# 6.101 recitation: iterators and generators

x = [9, 8, 7, 6, 5]

for i in x:
    print(i)


# generators using other generators!!!
# using generators, we can mimic the behaviors of a lot of built-in things (and
# others!)


def my_enumerate(x):
    ix = 0
    for elt in x:
        yield ix, elt
        ix += 1


def my_zip(x, y):
    x = iter(x)
    y = iter(y)
    while True:
        try:
            a = next(x)
            b = next(y)
        except StopIteration:
            return

        yield (a, b)


def interleave(x, y):
    x = iter(x)
    y = iter(y)
    while True:
        got_a = True
        try:
            a = next(x)
        except StopIteration:
            a = None
            got_a = False

        got_b = True
        try:
            b = next(y)
        except StopIteration:
            b = None
            got_b = False

        if got_a:
            yield a
        if got_b:
            yield b

        if not (got_a or got_b):
            return  # no a or b val, quit!


def my_reversed(x):
    for i in range(len(x) - 1, -1, -1):
        yield x[i]


# generators can also be recursive!  consider the following code, which is
# similar to code we've written a few times in 6.101 already:


def flatten(x):
    # here, x is an arbitrarily-nested list of numbers, i.e., a list of numbers
    # or other lists (when themselves contain numbers or other lists, and so
    # on).  we want to return a single flat list containing these numbers, but
    # with all nesting removed.
    out = []
    for elt in x:
        if isinstance(elt, list):
            # for i in flatten(elt):
            #    out.append(i)
            out.extend(flatten(elt))
        else:
            out.append(elt)
    return out


# if, instead, we wanted a _generator_ that yielded all of these values, we
# could do that like so (note that the 'yield from' line is effectively the
# same as the commented-out lines that precede it)


def flatten(x):
    for elt in x:
        if isinstance(elt, list):
            # for i in flatten(elt):
            #    yield i
            yield from flatten(elt)
        else:
            yield elt
