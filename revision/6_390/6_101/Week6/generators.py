import time


def list_range(start, stop, step=1):
    assert step >= 1
    out = []
    current = start
    while current < stop:
        out.append(current)
        current += step
    # print("Out",out)
    return out  # Out goes until the end of range


def gen_range(start, stop, step=1):  # Using Generators
    assert step >= 1
    current = start
    while current < stop:  # Loop Stops at the condition
        yield current
        current += step


s1 = time.perf_counter()
gen = gen_range(0, 1_000_000_000, step=1)
for val in gen:
    if val > 100:
        break
    print(val**2, end=" ")
s2 = time.perf_counter()
print("\n", s2 - s1)  # 0.0001 seconds

a = list_range(0, 1_000_000_000, step=1)
for val in a:
    if val > 100:
        break
    print(val**2, end=" ")
s3 = time.perf_counter()
print("\n", s3 - s2)  # 25 Seconds


def gen1():
    yield 1
    yield 2


def gen2():
    yield 8
    yield 9


def gen3():
    yield from gen1()
    yield from gen2()


def get_generators():
    # yield from gen1()
    # yield from gen2()
    yield from gen3()  # Yield From a function who returns a generator


for val in get_generators():
    print(val, end=" ")
print()

print(list(gen3()))


def negate_elements(x):
    if not x:
        return
    yield -x[0]
    yield from negate_elements(x[1:])


a = [1, 2, 3]
for val in negate_elements(a):
    print(val, end=" ")
