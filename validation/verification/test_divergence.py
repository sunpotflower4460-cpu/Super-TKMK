from validation.verification.divergence_test import run_verification


def test_float64_divergence_target_is_met():
    result = run_verification()
    assert result["max_abs"] <= result["target"]
    assert result["l2_norm"] <= result["target"]
