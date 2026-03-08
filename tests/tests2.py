import numpy as np
import pytest

from task import detect_anomalies


def test_basic_anomalies():
    data = np.array([
        21.0, 21.2, 21.3, 21.1, 21.4,
        21.2, 21.3, 50.0, 21.2, 21.3,
        21.1, 20.9, -10.0, 21.2
    ])
    result = detect_anomalies(data, 3)
    assert np.array_equal(result, np.array([7, 12]))


def test_no_anomalies():
    data = np.array([20.1, 20.2, 20.0, 20.3, 20.2, 20.1])
    result = detect_anomalies(data, 3)
    assert result.size == 0


def test_single_anomaly():
    data = np.array([10, 10, 10, 10, 100])
    result = detect_anomalies(data, 2)
    assert np.array_equal(result, np.array([4]))


def test_empty_array():
    data = np.array([])
    result = detect_anomalies(data, 3)
    assert result.size == 0


def test_sigma_zero():
    data = np.array([5, 5, 5, 5, 5])
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
        detect_anomalies(data, 2)


def test_threshold_negative():
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        detect_anomalies(data, -1)


def test_threshold_zero():
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        detect_anomalies(data, 0)
