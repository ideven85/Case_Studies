import dis


# Example 1: A Closure
def outer_function(factor):
    def inner_function(number):
        return number * factor  # 'factor' is a free variable

    return inner_function


# @outer_function
# def multiply_by_two(n):
#     print(n)
#


def make_multiplier_of(n):
    """Outer function that creates a closure."""

    def multiplier(x):
        """The inner function (closure) that remembers 'n'."""
        return x * n

    return multiplier


# Create a closure where 'n' is fixed at 3
times3 = make_multiplier_of(3)

# Disassemble the 'multiplier' function object
import dis

print("--- Disassembly of closure 'times3' ---")
dis.dis(times3)
