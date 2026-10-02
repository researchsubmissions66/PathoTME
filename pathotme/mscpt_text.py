"""RN50 MSCPT deep-text port; native 64-position consumption, frozen towers."""
import torch
from torch import nn
from torch.nn import functional as F


def rn50_layer(block, x):
    """Run the native residual block with a local causal mask, without mutation."""
    value = block.ln_1(x)
    mask = block.attn_mask[:len(x), :len(x)].to(device=x.device, dtype=x.dtype)
    x = x + block.attn(value, value, value, need_weights=False, attn_mask=mask)[0]
    return x + block.mlp(block.ln_2(x))


def token_ids(strings):
    from clip import tokenize
    return tokenize(strings, context_length=64, truncate=True)


class RN50TextTower(nn.Module):
    def __init__(self, raw, n_tpro=2, n_high=10):
        super().__init__()
        self.transformer = raw.transformer.float().requires_grad_(False)
        self.ln_final = raw.ln_final.float().requires_grad_(False)
        self.register_buffer('projection', raw.text_projection.detach().float().clone())
        self.n_tpro, self.n_high = n_tpro, n_high

    def forward(self, x, p_ins, p_uni, ids, attention_mask=None, causal_mask=None):
        layers, count, width = p_ins.shape
        classes = count // self.n_high
        contexts = p_ins.reshape(layers, classes, self.n_high, width)
        contexts = contexts.unsqueeze(2).expand(-1, -1, len(x)//classes, -1, -1)
        contexts = contexts.reshape(layers, len(x), self.n_high, width)
        for index, block in enumerate(self.transformer.resblocks):
            if index:
                x = torch.cat((x[:, :1], p_uni[index-1].unsqueeze(0).expand(len(x), -1, -1),
                               contexts[index-1], x[:, 1+self.n_tpro+self.n_high:]), dim=1)
            x = rn50_layer(block, x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x)
        return x[torch.arange(len(x), device=x.device), ids.argmax(-1)] @ self.projection


class RN50PromptLearner(nn.Module):
    """Retain MSCPT's shared and description-derived layer prompts at width 512."""
    def __init__(self, raw, bank, n_tpro=2, n_high=10):
        super().__init__()
        self.n_tpro, self.n_high = n_tpro, n_high
        self.layers, width = len(raw.transformer.resblocks), raw.ln_final.weight.numel()
        prefix = ' '.join(['X']*(n_tpro+n_high))
        strings = [f'{prefix} {text}' for branch in bank.values() for text in branch['big_mag']]
        ids = token_ids(strings).to(raw.positional_embedding.device)
        if not (ids.argmax(-1) > n_tpro+n_high).all():
            raise ValueError('MSCPT prompt slots consume EOT')
        with torch.no_grad():
            embeddings = raw.token_embedding(ids).float() + raw.positional_embedding[:64].float()
        self.register_buffer('tokenized_prompts', ids)
        self.register_buffer('prefix', embeddings[:, :1].clone())
        self.register_buffer('suffix', embeddings[:, 1+n_tpro+n_high:].clone())
        self.p_input = nn.Parameter(torch.empty(n_tpro+n_high, width))
        self.p_uni = nn.ParameterList([nn.Parameter(torch.empty(n_tpro, width)) for _ in range(self.layers-1)])
        for value in [self.p_input, *self.p_uni]: nn.init.normal_(value, std=0.02)
        self.p_ins_projector = nn.Linear(width, width)

    def forward(self, feats, desc):
        x = torch.cat((self.prefix, self.p_input.unsqueeze(0).expand(len(self.prefix), -1, -1), self.suffix), 1)
        feats = feats.permute(1, 0, 2, 3)
        feats = feats.reshape(feats.shape[0], -1, feats.shape[-1])[:self.layers-1]
        contexts = feats.float() + self.p_ins_projector(feats.float())
        return x, contexts, self.p_uni, None, None


@torch.no_grad()
def frozen_rn50(raw, strings):
    ids = token_ids(strings).to(raw.positional_embedding.device)
    x = raw.token_embedding(ids).float() + raw.positional_embedding[:64].float()
    indices = torch.arange(len(ids), device=ids.device)
    layers = []
    for block in raw.transformer.resblocks:
        x = rn50_layer(block, x.permute(1,0,2)).permute(1,0,2)
        layers.append(x[indices, ids.argmax(-1)])
    result = raw.ln_final(x)[indices, ids.argmax(-1)] @ raw.text_projection.float()
    return F.normalize(result, dim=-1), F.normalize(torch.stack(layers), dim=-1)
