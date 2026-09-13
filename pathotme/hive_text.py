"""Frozen native PLIP/RN50 text towers with HiVE's shared 16-token prefix."""
from collections import OrderedDict
import hashlib
import torch
from torch import nn


def ordered_strings(bank):
    prefix = ' '.join(['X'] * 16)
    return [f'{prefix} {term} : {explanation}'
            for branch in bank.values() for term, explanation in branch.items()]


class HiVETokenPromptLearner(nn.Module):
    def __init__(self, embeddings, ids):
        super().__init__()
        if embeddings.shape[:2] != ids.shape or ids.shape != (32,77):
            raise ValueError('two classes, sixteen descriptions each, native context 77 required')
        if not (ids.argmax(-1) > 16).all():
            raise ValueError('native EOT must follow the sixteen context slots')
        self.ctx = nn.Parameter(embeddings.new_empty(16,embeddings.shape[-1]))
        nn.init.normal_(self.ctx, std=0.02)
        self.register_buffer('token_prefix', embeddings[:,:1].detach().clone())
        self.register_buffer('token_suffix', embeddings[:,17:].detach().clone())
        self.register_buffer('tokenized_prompts', ids.detach().clone())

    def forward(self, device=None):
        return torch.cat((self.token_prefix, self.ctx.unsqueeze(0).expand(32,-1,-1),
                          self.token_suffix), dim=1)


class HiVENativeTextEncoder(nn.Module):
    """Encode a flat ordered bank and restore HiVE's class-keyed dictionary."""
    def __init__(self, raw, encoder, class_names, mask=None):
        super().__init__()
        self.encoder = encoder
        self.class_names = tuple(class_names)
        if encoder == 'plip':
            self.text_model = raw.text_model.float().requires_grad_(False)
            self.text_projection = raw.text_projection.float().requires_grad_(False)
            self.register_buffer('attention_mask', mask.detach().clone())
        elif encoder == 'clip-rn50':
            self.transformer = raw.transformer.float().requires_grad_(False)
            self.ln_final = raw.ln_final.float().requires_grad_(False)
            self.register_buffer('positional_embedding', raw.positional_embedding.detach().float().clone())
            self.register_buffer('text_projection', raw.text_projection.detach().float().clone())
        else:
            raise ValueError('unsupported HiVE text tower')

    def forward(self, prompts, ids):
        eot = ids.argmax(-1)
        if self.encoder == 'plip':
            handle = self.text_model.embeddings.token_embedding.register_forward_hook(
                lambda _module,_inputs,_output: prompts)
            try:
                output = self.text_model(input_ids=ids, attention_mask=self.attention_mask)
            finally:
                handle.remove()
            hidden = output.last_hidden_state
            features = self.text_projection(hidden[torch.arange(len(ids),device=ids.device), eot])
        else:
            hidden = prompts.float() + self.positional_embedding
            hidden = self.transformer(hidden.permute(1,0,2)).permute(1,0,2)
            hidden = self.ln_final(hidden)
            features = hidden[torch.arange(len(ids),device=ids.device), eot] @ self.text_projection
        # HiVE uses unnormalized paired text features before graph normalization.
        return OrderedDict((name,features[i*16:(i+1)*16]) for i,name in enumerate(self.class_names))


def build_text_port(encoder, bank, expected_token_sha256=None):
    raw = encoder.raw_model.float().eval().requires_grad_(False)
    device = next(raw.parameters()).device
    strings = ordered_strings(bank)
    if encoder.spec.name == 'plip':
        if int(raw.config.text_config.max_position_embeddings) != 77:
            raise ValueError('unexpected PLIP context')
        tokens = encoder.raw_tokenizer(strings, padding='max_length', truncation=True,
            max_length=77, return_tensors='pt')
        ids, mask = tokens['input_ids'].to(device), tokens['attention_mask'].to(device)
        with torch.no_grad(): embeddings = raw.text_model.embeddings.token_embedding(ids)
    else:
        # Explicit native prefix/EOT policy, including all sixteen context slots.
        from clip import tokenize
        ids = tokenize(strings, context_length=77, truncate=True).to(device)
        mask = None
        with torch.no_grad(): embeddings = raw.token_embedding(ids)
    if not torch.equal(ids.argmax(-1), ids.eq(49407).long().argmax(-1)):
        raise ValueError('paired CLIP EOT pooling mismatch')
    digest = hashlib.sha256(ids.detach().cpu().numpy().astype('<i8').tobytes()).hexdigest()
    if expected_token_sha256 is not None and digest != expected_token_sha256:
        raise ValueError('native prompt token consumption changed after preparation')
    return (HiVETokenPromptLearner(embeddings,ids),
            HiVENativeTextEncoder(raw,encoder.spec.name,list(bank),mask))
