import pytest
import numpy as np
from task import minmax_scale


def test_basic_case():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    result = minmax_scale(X)
    expected = np.array([[0, 0], [0.5, 0.5], [1, 1]])
    assert np.allclose(result, expected)


def test_constant_column():
    X = np.array([[2, 3], [2, 5], [2, 7]])
    result = minmax_scale(X)
    expected = np.array([[0, 0], [0, 0.5], [0, 1]])
    assert np.allclose(result, expected)


def test_float_values():
    X = np.array([[0.5, 1.0], [1.5, 2.0]])
    result = minmax_scale(X)
    expected = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert np.allclose(result, expected)


def test_empty_array():
    X = np.array([])
    with pytest.raises(ValueError):
        minmax_scale(X)


def test_non_numpy_input():
    X = [[1, 2], [3, 4]]
    with pytest.raises(TypeError):
        minmax_scale(X)


def test_non_2d_array():
    X1 = np.array([1, 2, 3])
    X2 = np.array([[[1, 2], [3, 4]]])
    with pytest.raises(ValueError):
        minmax_scale(X1)
    with pytest.raises(ValueError):
        minmax_scale(X2)


def test_single_element():
    X = np.array([[5]])
    result = minmax_scale(X)
    expected = np.array([[0]])
    assert np.allclose(result, expected)


def test_zero_range_column():
    X = np.array([[7, 1], [7, 4], [7, 7]])
    result = minmax_scale(X)
    expected = np.array([[0, 0], [0, 0.5], [0, 1]])
    assert np.allclose(result, expected)


def test_negative_values():
    X = np.array([[-5, 0], [0, 10], [5, 20]])
    result = minmax_scale(X)
    expected = np.array([[0, 0], [0.5, 0.5], [1, 1]])
    assert np.allclose(result, expected)
