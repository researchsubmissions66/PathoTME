import numpy as np

from pathotme.fusion import blend_probabilities, probability_to_log_odds


def test_fusion_endpoints_recover_each_modality():
    vila = np.array([0.1, 0.8, 0.6])
    tme = np.array([0.7, 0.2, 0.9])
    np.testing.assert_allclose(blend_probabilities(vila, tme, 0), vila)
    np.testing.assert_allclose(blend_probabilities(vila, tme, 1), tme)


def test_equal_weight_is_mean_in_log_odds_space():
    vila = np.array([0.2, 0.7])
    tme = np.array([0.8, 0.4])
    fused = blend_probabilities(vila, tme, 0.5)
    np.testing.assert_allclose(
        probability_to_log_odds(fused),
        (probability_to_log_odds(vila) + probability_to_log_odds(tme)) / 2,
        atol=1e-15,
    )
