import math


def sine(period):
    n = 0
    print(n)
    while True:

        yield round(math.sin(n * 2 * math.pi / period), 6)
        n = (n + 1) % period


a = sine(5)
for val in a:
    print(val)
