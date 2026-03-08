import pytest
import numpy as np
from task import max_pooling


def test_basic_case():
    x = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    result = max_pooling(x, pool_size=2)
    expected = np.array([[6, 8],
                         [14, 16]])
    assert np.array_equal(result, expected)


def test_non_divisible_size():
    x = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    result = max_pooling(x, pool_size=2)
    expected = np.array([[5]])
    assert np.array_equal(result, expected)


def test_pool_size_one():
    x = np.array([[1, 2],
                  [3, 4]])
    result = max_pooling(x, pool_size=1)
    expected = np.array([[1, 2],
                         [3, 4]])
    assert np.array_equal(result, expected)


def test_pool_size_larger_than_matrix():
    x = np.array([[1, 2],
                  [3, 4]])
    result = max_pooling(x, pool_size=3)
    expected = np.empty((0,0))
    assert result.shape == expected.shape


def test_non_numpy_input():
    x = [[1, 2], [3, 4]]
    with pytest.raises(TypeError):
        max_pooling(x, pool_size=2)


def test_non_2d_array():
    x1 = np.array([1, 2, 3])
    x2 = np.array([[[1, 2], [3, 4]]])
    with pytest.raises(ValueError):
        max_pooling(x1, pool_size=2)
    with pytest.raises(ValueError):
        max_pooling(x2, pool_size=2)


def test_invalid_pool_size():
    x = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        max_pooling(x, pool_size=0)
    with pytest.raises(ValueError):
        max_pooling(x, pool_size=-1)


def test_negative_numbers():
    x = np.array([[ -1, -2],
                  [ -3, -4]])
    result = max_pooling(x, pool_size=2)
    expected = np.array([[-1]])
    assert np.array_equal(result, expected)


def test_single_element():
    x = np.array([[7]])
    result = max_pooling(x, pool_size=1)
    expected = np.array([[7]])
    assert np.array_equal(result, expected)
