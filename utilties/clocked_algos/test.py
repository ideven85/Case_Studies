import pytest
from utilties import clock_utils as c

# from clock_utils import clock_deco
from .calc_fib import fib

clock_deco = c.clock_deco


# @clock_deco


def calc_fib(n):
    if n < 2:
        return n
    else:
        a, b = 0, 1
        for _ in range(1, n):
            a, b = b, a + b
    return b


# @pytest.param(range(1,100),n)
def test_fib_small():
    assert fib(10) == calc_fib(10)


@pytest.mark.parametrize("num,output", [(1, 11)])
def test_multiplication_by_11(num, output):
    assert 11 * num == output


def test_fib_large():
    assert fib(100) == calc_fib(100)


def main():
    print(f":test fib small:{test_fib_small()}$")
    test_fib_small()
    test_fib_large()


if __name__ == "__main__":
    main()
