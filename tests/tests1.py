import pytest
import numpy as np
from task import weighted_average_price


def test_basic_case():
    prices = np.array([10, 20, 30])
    quantities = np.array([2, 3, 5])
    result = weighted_average_price(prices, quantities)
    expected = (10*2 + 20*3 + 30*5) / (2+3+5)
    assert result == expected

def test_float_values():
    prices = np.array([10.5, 20.2, 30.1])
    quantities = np.array([1.5, 2.0, 3.5])
    result = weighted_average_price(prices, quantities)
    expected = np.sum(prices * quantities) / np.sum(quantities)
    assert np.isclose(result, expected)

def test_zero_quantity_error():
    prices = np.array([10, 20])
    quantities = np.array([0, 0])
    with pytest.raises(ValueError):
        weighted_average_price(prices, quantities)

def test_different_lengths_error():
    prices = np.array([10, 20, 30])
    quantities = np.array([1, 2])
    with pytest.raises(ValueError):
        weighted_average_price(prices, quantities)

def test_non_1d_array_error():
    prices = np.array([[10, 20], [30, 40]])
    quantities = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError):
        weighted_average_price(prices, quantities)

    prices = np.array([10, 20, 30])
    quantities = np.array([[1, 2, 3]])
    with pytest.raises(ValueError):
        weighted_average_price(prices, quantities)

def test_non_numpy_input_error():
    prices = [10, 20, 30]  # не numpy
    quantities = np.array([1, 2, 3])
    with pytest.raises(TypeError):
        weighted_average_price(prices, quantities)

    prices = np.array([10, 20, 30])
    quantities = [1, 2, 3]  # не numpy
    with pytest.raises(TypeError):
        weighted_average_price(prices, quantities)

def test_all_zeros_prices():
    prices = np.array([0, 0, 0])
    quantities = np.array([1, 2, 3])
    result = weighted_average_price(prices, quantities)
    assert result == 0

def test_single_element():
    prices = np.array([50])
    quantities = np.array([5])
    result = weighted_average_price(prices, quantities)
    assert result == 50
