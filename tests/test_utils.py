from functools import partial

import pytest

from miniai.utils import compose, listify


@pytest.mark.parametrize("test_input, expected", [([1, 2, 3], [1, 2, 3])])
def test_listify(test_input, expected):
    assert listify(test_input) == expected


def power(a, b):
    return a**b


power_two = partial(power, b=2)
power_three = partial(power, b=3)


@pytest.mark.parametrize(
    "test_input, funcs, expected", [(2, [power_two, power_three], 64)]
)
def test_compose(test_input, funcs, expected):
    assert compose(test_input, funcs) == expected
