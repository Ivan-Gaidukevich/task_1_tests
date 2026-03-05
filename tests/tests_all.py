import ast
import numpy as np
import pytest
from task import trim_mean_dive


def test_numpy_usage():
    with open("task_1/task1_.py", "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    numpy_used = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if node.value.id in ("np", "numpy"):
                    numpy_used = True
    assert numpy_used, "В решении должен использоваться numpy"


# Формат: (points, hard_coef, expected, score)
hidden_tests = [
    # 5 судей
    (np.array([7, 9, 6, 5, 8]), None, 7.0, 2),
    (np.array([10, 10, 9, 8, 7]), None, 9.0, 2),
    (np.array([0, 0, 1, 0, 0]), None, 0.0, 2),
    (np.array([8, 8, 8, 8, 8]), None, 8.0, 2),  # одинаковые оценки

    # 7 судей
    (np.array([6, 7, 8, 5, 9, 6, 7]), 2.0, 34.0 * 2.0 * 0.6, 3),
    (np.array([8, 7, 9, 6, 10, 8, 7]), 3.0, 39 * 3 * 0.6, 3),
    (np.array([5, 5, 5, 5, 5, 5, 5]), 1.5, 25 * 1.5 * 0.6, 3),
    (np.array([10, 10, 10, 10, 10, 10, 10]), 1.2, 50 * 1.2 * 0.6, 3),  # максимум
    (np.array([0, 0, 0, 0, 0, 0, 0]), 1.2, 0.0, 3),  # минимум
]


@pytest.mark.parametrize("points,hard_coef,expected,score", hidden_tests)
def test_hidden_cases(points, hard_coef, expected, score, request):
    if hard_coef is None:
        result = trim_mean_dive(points)
    else:
        result = trim_mean_dive(points, hard_coef)
    assert round(result, 2) == round(expected, 2)

    if not hasattr(request.config, "total_score"):
        request.config.total_score = 0
    request.config.total_score += score


@pytest.mark.parametrize("points", [
    np.array([7, 8, 9]),  # меньше 5 судей
    np.array([6, 7, 8, 9, 10, 5, 6, 7]),  # больше 7 судей
])
def test_invalid_number_of_judges(points):
    with pytest.raises(ValueError):
        trim_mean_dive(points)


@pytest.mark.parametrize("points", [
    np.array([-1, 7, 8, 9, 6]),  # оценка < 0
    np.array([11, 7, 8, 9, 6]),  # оценка > 10
])
def test_invalid_score_range(points):
    with pytest.raises(ValueError):
        trim_mean_dive(points)


@pytest.mark.parametrize("points,hard_coef", [
    (np.array([7, 8, 6, 9, 5, 8, 7]), 1.0),  # коэффициент < 1.2
    (np.array([7, 8, 6, 9, 5, 8, 7]), 4.0),  # коэффициент > 3.6
])
def test_invalid_hard_coef(points, hard_coef):
    with pytest.raises(ValueError):
        trim_mean_dive(points, hard_coef)
