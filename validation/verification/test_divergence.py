import pytest

from validation.verification.divergence_test import run_verification


def test_float64_divergence_target_is_met():
    result = run_verification(dtype_name="float64")
    assert result["max_abs"] <= result["target"]
    assert result["l2_norm"] <= result["target"]


def test_cupy_backend_requires_cupy_installation(monkeypatch):
    from validation.verification import divergence_test

    monkeypatch.setattr(divergence_test, "cp", None)
    with pytest.raises(RuntimeError, match="CuPy not found"):
        divergence_test.get_backend("cupy")
