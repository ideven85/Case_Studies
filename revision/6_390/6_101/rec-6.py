# 6.101 recitation:
#   * functions as first-class objects
#   * higher-order functions


############### Creating and calling a function


def foo(x):
    return 3 * x**4


a = foo(3)


############### Taking derivative of a function


def foo(x):
    return 3 * x**4


def deriv(f, dx):
    """
    Given a function f of one variable (which takes a float value and returns a
    float value) returns a function that approximates the derivative df/dx.

    dx (assumed positive) > 0 is the width of the approximation (the true
    derivative is the limit as dx -> 0).
    """
    raise NotImplementedError


dfoo = deriv(foo, 0.001)

print(dfoo(5))


print(deriv(dfoo)(5))


############### nth derivatives


def nth_derivative(f, n, dx):
    """
    Given a function f of one variable (takes a float value and returns a float value)
    returns a function that approximates the nth derivative of f using approximation width dx.
    """


h = nth_derivative(lambda x: x**5, 3, 0.001)  # => third derivative of x^5 = 60x^2
