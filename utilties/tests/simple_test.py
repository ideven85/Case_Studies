import pytest


@pytest.fixture
def input_value():
    # input = 39
    return 39


@pytest.fixture
def divisible_by_3(input_value):
    return input_value % 3 == 0


@pytest.fixture
def divisible_by_6(input_value):
    return input_value % 6 == 0


@pytest.mark.parametrize("num, output", [(1, 11), (2, 22), (3, 35), (4, 44)])
def test_multiplication_11(num, output):
    assert 11 * num == output


def test_divisible_by_3_or_6(divisible_by_3, divisible_by_6):
    assert divisible_by_6 or divisible_by_3
