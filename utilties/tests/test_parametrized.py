import pytest


@pytest.mark.parametrize("a, b", [(1, 11)])
def test_multiplication(a, b):
    assert 11 * a == b


@pytest.mark.parametrize("a,b", [(1, 2)])
def test_addition(a, b):
    assert 1 + a == b
