import time
import sys

sys.setrecursionlimit(10000)


def original_sum_list(x):
    if not x:
        return 0
    return x[0] + original_sum_list(x[1:])


def sum_list_iterative(x):
    sum_so_far = 0
    for element in x:
        sum_so_far += element
    return sum_so_far


count = 0


def factorial_recursive(n):
    return 1 if n < 2 else n * factorial_recursive(n - 1)


def factorial_iterative(n):
    out = 1
    for i in range(1, n + 1):
        out *= i
    return out


def sum_list(x):
    count = 0

    def sum_list_helper(sum_so_far, lst):
        nonlocal count
        if not lst:
            return sum_so_far

        first = lst[0]
        rest = lst[1:]
        count += 1
        return sum_list_helper(sum_so_far + first, rest)

    a = sum_list_helper(0, x)
    print(count)
    return a


def sum_listV2(x, i=0, sum_so_far=0):
    if i >= len(x):
        return sum_so_far
    return sum_listV2(x, i + 1, sum_so_far + x[i])


def sum_nested(original_x):
    sum_so_far = 0
    agenda = [original_x]

    while agenda:  # O(n)
        x = agenda.pop(-1)
        if not x:
            continue
        elif isinstance(x[0], list):
            agenda.append(x[0])
            agenda.append(x[1:])  # O(n)
        else:
            sum_so_far += x[0]
            agenda.append(x[1:])  # O(n)
    return sum_so_far


def sum_nested_recursive(original_x):
    if not original_x:
        return 0
    elif isinstance(original_x[0], list):
        return sum_nested_recursive(original_x[0]) + sum_nested_recursive(
            original_x[1:]
        )
    else:
        return original_x[0] + sum_nested_recursive(original_x[1:])


# def sum_list2(x):
#     if not x:
#         return 0
#     else:
#         m = len(x) // 2
#         return sum_list2(x[0:m]) + sum_list2(x[m:])


if __name__ == "__main__":
    l = list(range(1, 102))

    # print("Sum List V2")
    # start = time.time()
    # print(sum_listV2(l))
    # end = time.time()
    # print((end - start) * 1000000)
    # print("Sum List Iterative")
    #
    # start = time.time()
    # print(sum_list_iterative(l))
    # end = time.time()
    # print((end - start) * 1000000)
    #
    # print("Sum List Helper")
    # start = time.time()
    # print(sum_list(l))
    # end = time.time()
    # print((end - start) * 1000000)
    #
    # print("Original Sum List")
    # start = time.time()
    # print(original_sum_list(l))
    # end = time.time()
    # print((end - start) * 1000000)

    a = [1, [2, 3], [[list(range(4, 100))]], 101]

    print("Iterative Nested Sum List")
    start = time.time_ns()
    print(sum_nested(a))
    end = time.time_ns()
    print((end - start))
    print(sum_list([1, 2, 3, 4]))

    print("Recursive Nested Sum List")

    start1 = time.time_ns()
    print(sum_nested_recursive(a))
    end1 = time.time_ns()
    print((end1 - start1))

    # print(factorial_recursive(1001))
    print(factorial_iterative(1001))
