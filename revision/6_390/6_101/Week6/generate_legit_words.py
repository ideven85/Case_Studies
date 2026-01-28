import time
from multiprocessing.pool import ThreadPool
from threading import Thread
import functools

with open("../Week4/all_words.txt", "r") as f:
    ALL_WORDS = {i.strip() for i in f}
# print(all_words[:20])
# print("" in ALL_WORDS)


def generate_subsequences_slow(word):
    return {w for w in generate_subsequences_slow_helper(word) if w in ALL_WORDS}


def generate_subsequences_fast(word):
    # z=[w for w in generate_subsequences_fast_helper(word) if w in ALL_WORDS]
    return [w for w in generate_subsequences_fast_helper(word) if w in ALL_WORDS]

    # print(len(out))
    # return out
    # return {w for w in generate_subsequences_fast_helper(word) if w in ALL_WORDS}


def generate_subsequences_fast_helper(word):
    if not word:
        yield ""
        return

    first = word[0]
    rest = word[1:]

    for w in generate_subsequences_fast_helper(rest):
        yield w
        for index in range(len(w) + 1):
            yield w[:index] + first + w[index:]


def generate_subsequences_slow_helper(word):
    if not word:
        return {""}
    first = word[0]
    rest = word[1:]
    out = set()
    for w in generate_subsequences_slow_helper(rest):
        out.add(w)
        for index in range(len(w) + 1):
            out.add(w[:index] + first + w[index:])
    return out


def subsequences(word):
    out = [[]]
    for char in word:
        # I need more practice here
        for i in range(len(out)):
            current = out[i]
            out.append(current + [char])
    return out


if __name__ == "__main__":
    words = [
        "fib",
        "fie",
        "fig",
        "fin",
        "fir",
        "fit",
        "fix",
        "cheer",
    ]
    z = subsequences("artichokes")
    sub = []
    for el in z:
        sub.append("".join(el))
    print(sub, len(sub), "\n", len([w for w in sub if w in ALL_WORDS]))
    # [generate_subsequences_fast(w) for w in words]
    # start = time.time_ns()
    # with ThreadPool() as pool:
    #     for result in pool.map(generate_subsequences_fast, [w for w in words]):
    #         print(f"Got Result {result}")
    # end = time.time_ns()
    # print("\n\n\nDone", end - start)
    # threads = [Thread(target=generate_subsequences_fast,args=(x,)) for x in words]
    # start = time.time_ns()
    # for t in threads:
    #     t.start()
    # for t in threads:
    #     t.join()
    # end = time.time_ns()
    # print('\nDone',end-start)

    #
    # start = time.perf_counter()
    # print(generate_subsequences_slow("artichokes"))
    # end = time.perf_counter()
    # print(end - start)
    # print(end - start)
    #
    # """
    # Using Generators
    # """
    start = time.time()
    s = set(sub)
    sub2 = generate_subsequences_fast("artichokes")
    print(s.difference(sub2))
    # input(len(sub2))
    # assert len(z) == len(sub2)

    print()
    end = time.time()
    print(end - start)
"""Step 1: Highlight the function below"""


def add_one(x):
    return lambda y: x + y + 1


"""Step 2: Create a closure using `add_one` and store it in `plus_two`"""
plus_two = add_one(2)


# Factorial function using reduce
def factorialV2(n):
    return 1 if n < 2 else functools.reduce(lambda a, b: a * b, range(2, n + 1))


print(factorialV2(5))
