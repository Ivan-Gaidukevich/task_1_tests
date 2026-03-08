import numpy as np
import pytest

from task import detect_anomalies


def test_basic_anomalies_pass():
    data = np.array([10, 10, 10, 10, 50, 10, 10])
    result = detect_anomalies(data, threshold)
    assert np.array_equal(result, np.array([4]))
    

def test_multiple_anomalies_pass():
    data = np.array([5, 5, 5, 50, 5, 5, -20])
    threshold = 2
    result = detect_anomalies(data, threshold)
    assert np.array_equal(result, np.array([3, 6]))


def test_single_anomaly_high_threshold():
    data = np.array([1, 1, 1, 10, 1])
    threshold = 1.5
    result = detect_anomalies(data, threshold)
    assert np.array_equal(result, np.array([3]))


def test_edge_case_anomaly_at_end():
    data = np.array([2, 2, 2, 2, 2, 20])
    threshold = 1.5
    result = detect_anomalies(data, threshold)
    assert np.array_equal(result, np.array([5]))


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
