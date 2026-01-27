# from utilties.clock_utils import clock_deco
import time
import sys

sys.setrecursionlimit(100000)
from .. import utils, clock_utils

memoization = utils.memoization
clock_deco = clock_utils.clock_deco


@memoization
@clock_deco
def fib(n):
    """
    @param n: n int
    @returns : fibonacci of n


    """
    return n if n < 2 else fib(n - 1) + fib(n - 2)


@clock_deco
def sleep(seconds):
    time.sleep(seconds)


@clock_deco
@memoization
def factorial(n):
    """
    Computes the factorial of a non negative number
    :param n: n int non negative integer
    :return: the factorial of the number
    """
    return 1 if n < 2 else n * factorial(n - 1)


sleep(10)
