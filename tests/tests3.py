import pytest
import numpy as np
from task.py import moving_average  


def test_basic_case():
    data = np.array([1, 2, 3, 4, 5])
    window = 3
    result = moving_average(data, window)
    expected = np.array([2.0, 3.0, 4.0])
    assert np.allclose(result, expected)


def test_window_one():
    data = np.array([10, 20, 30])
    window = 1
    result = moving_average(data, window)
    expected = data
    assert np.allclose(result, expected)


def test_window_equals_length():
    data = np.array([1, 2, 3, 4])
    window = 4
    result = moving_average(data, window)
    expected = np.array([2.5])
    assert np.allclose(result, expected)


def test_float_values():
    data = np.array([0.5, 1.5, 2.5, 3.5])
    window = 2
    result = moving_average(data, window)
    expected = np.array([1.0, 2.0, 3.0])
    assert np.allclose(result, expected)


def test_empty_array():
    data = np.array([])
    window = 1
    with pytest.raises(ValueError):
        moving_average(data, window)


def test_non_numpy_input():
    data = [1, 2, 3]
    window = 2
    with pytest.raises(TypeError):
        moving_average(data, window)


def test_non_1d_array():
    data = np.array([[1, 2], [3, 4]])
    window = 2
    with pytest.raises(ValueError):
        moving_average(data, window)


def test_zero_or_negative_window():
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        moving_average(data, 0)
    with pytest.raises(ValueError):
        moving_average(data, -1)


def test_window_larger_than_data():
    data = np.array([1, 2, 3])
    window = 5
    with pytest.raises(ValueError):
        moving_average(data, window)


def test_small_array_window_two():
    data = np.array([4, 6])
    window = 2
    result = moving_average(data, window)
    expected = np.array([5.0])
    assert np.allclose(result, expected)
