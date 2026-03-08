import numpy as np
import pytest
from task import detect_anomalies


def test_basic_anomaly():
    data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100])
    result = detect_anomalies(data, 2)
    assert np.array_equal(result, np.array([12]))


def test_multiple_anomalies():
    data = np.array([99, 1, 1, 1, 1, 1, 99, 1, 1, 1, 1, 1, 100])
    result = detect_anomalies(data, 3)
    assert np.array_equal(result, np.array([0, 6, 12]))


def test_empty_array():
    data = np.array([])
    result = detect_anomalies(data, 3)
    assert result.size == 0


def test_sigma_zero():
    data = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
    result = detect_anomalies(data, 3)
    assert result.size == 0


def test_data_not_numpy():
    with pytest.raises(TypeError):
        detect_anomalies([1, 2, 3], 2)


def test_threshold_not_number():
    data = np.array([1, 2, 3])
    with pytest.raises(TypeError):
        detect_anomalies(data, "3")


def test_data_not_1d():
    data = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        detect_anomalies(data, 3)


def test_threshold_negative():
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        detect_anomalies(data, -1)


def test_threshold_zero():
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        detect_anomalies(data, 0)
