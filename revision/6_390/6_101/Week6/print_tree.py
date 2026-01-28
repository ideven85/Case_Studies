import time

tree1 = [13, [7], [8, [99], [16, [77]], [42]]]


def all_values(tree):
    if not tree:
        return

    for el in tree:
        if isinstance(el, list):
            yield from all_values(el)
        else:
            yield el


def fib_gen(n):
    a = 0
    b = 1
    for i in range(n):
        yield a
        a, b = b, a + b


def fib(n):
    a = 0
    b = 1
    out1 = []
    for i in range(n):
        out1.append(a)
        a, b = b, a + b

    return out1


def main():

    x1 = fib_gen(10)
    s1 = time.time_ns()
    for val in x1:
        print(val, end=" ")
    s2 = time.time_ns()

    print(s2 - s1)

    for word in all_values(tree1):
        print(word, end=" ")
        print()
    s3 = time.perf_counter_ns()
    for i, val in fib_gen(1000):
        if i == 100:
            break
        print(val, end=" ")
    s4 = time.perf_counter_ns()
    print(s4 - s3)
