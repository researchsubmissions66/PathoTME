import numpy as np
import pytest
import torch

from pathotme.features import FEATURE_NAMES
from pathotme.guided_vila import (
    FoldStandardizer,
    SemanticTMETokenizer,
    TMEPrototypeConditioner,
)


def test_fold_standardizer_has_fixed_width_and_training_only_statistics():
    values = np.vstack([
        np.arange(len(FEATURE_NAMES), dtype=np.float32),
        np.arange(len(FEATURE_NAMES), dtype=np.float32) + 2,
        np.arange(len(FEATURE_NAMES), dtype=np.float32) + 4,
    ])
    values[0, 3] = np.nan
    standardizer = FoldStandardizer()
    standardizer.fit(values)
    transformed = standardizer(torch.from_numpy(values))
    assert transformed.shape == (3, 62)
    assert torch.isfinite(transformed).all()
    assert transformed.mean(dim=0).abs().max().item() < 1e-5


def test_semantic_tokenizer_builds_five_low_and_eleven_high_tokens():
    tokenizer = SemanticTMETokenizer(hidden_dim=16, dropout=0.0)
    low, high = tokenizer(torch.randn(2, len(FEATURE_NAMES)))
    assert low.shape == (5, 2, 16)
    assert high.shape == (11, 2, 16)
    assert SemanticTMETokenizer.LOW_TOKEN_NAMES[-1] == "tls"
    assert "cell_lymphocyte" in SemanticTMETokenizer.HIGH_TOKEN_NAMES
    assert "spatial_lymphocytes" in SemanticTMETokenizer.HIGH_TOKEN_NAMES


@pytest.mark.parametrize("scale,token_count", [("low", 5), ("high", 11)])
def test_conditioner_returns_prototype_aligned_attention(scale, token_count):
    torch.manual_seed(7)
    conditioner = TMEPrototypeConditioner(
        model_dim=32, hidden_dim=16, num_heads=4, dropout=0.0)
    centers = torch.randn(3, 1, 32, requires_grad=True)
    tme = torch.randn(1, len(FEATURE_NAMES))
    conditioned, attention = conditioner(centers, tme, scale=scale)
    assert conditioned.shape == centers.shape
    assert attention.shape == (1, 3, token_count)
    torch.testing.assert_close(
        attention.sum(dim=-1), torch.ones(1, 3), atol=1e-6, rtol=1e-6)
    conditioned.square().mean().backward()
    assert conditioner.output_projection.weight.grad is not None


def test_conditioned_centers_change_with_quantitative_values():
    torch.manual_seed(11)
    conditioner = TMEPrototypeConditioner(
        model_dim=32, hidden_dim=16, num_heads=4, dropout=0.0)
    centers = torch.randn(3, 1, 32)
    first, _ = conditioner(
        centers, torch.zeros(1, len(FEATURE_NAMES)), scale="high")
    second, _ = conditioner(
        centers, torch.ones(1, len(FEATURE_NAMES)), scale="high")
    assert not torch.allclose(first, second)
