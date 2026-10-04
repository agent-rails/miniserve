import pytest

from miniserve.bench.canary import measure_canary_ms, within_tolerance


def test_within_tolerance_boundaries():
    assert within_tolerance(20.0, 20.0)
    assert within_tolerance(20.0, 23.0)
    assert not within_tolerance(20.0, 23.1)
    assert within_tolerance(20.0, 10.0)


@pytest.mark.parametrize("baseline,value,tol", [(0.0, 1.0, 1.1), (1.0, 0.0, 1.1), (1.0, 1.0, 0.9)])
def test_within_tolerance_rejects_invalid_inputs(baseline, value, tol):
    with pytest.raises(ValueError):
        within_tolerance(baseline, value, tol)


def test_canary_returns_positive_milliseconds(cpu_model):
    assert measure_canary_ms(cpu_model) > 0
