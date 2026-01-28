import sys

sys.setrecursionlimit(1000)
# Maximum Recursive Calls is: 997


class SumList(list):

    def sum(self):
        if not self:
            return 0
        return self[0] + SumList(self[1:]).sum()


import doctest

print(doctest.testmod())


def sum_right(seq, sum_so_far=0, count=0):
    """
    Tail-call optimization can't be applied to a recursive call that isn't at the very end of the function.
     If sum_list were written as we originally had it, with return x[0] + sum_list(x[1:]), then this is not a tail call,
    Because the function still needs to do some more work (adding x[0]) after the recursive call comes back.
     Tail-call optimization is also blocked if the frame needs to be kept for a function object that was created during the call.


    """

    # if not seq:
    #     print("Count:",count)
    #     return sum_so_far
    # return sum_so_far+seq[0]+ sum_right(seq[1:],sum_so_far, count+1)
    """
    Using Tail Call Optimisation
    """
    if count >= len(seq):
        return sum_so_far
    return sum_right(seq, sum_so_far + seq[count], count + 1)


def until(n, filter_func, v):
    if n == v:
        return 0

    if filter_func(v):
        return sum([v]) + until(n, filter_func, v + 1)
    else:
        return until(n, filter_func, v + 1)


def sum_list(x):

    def sum_list_helper(sum_so_far, lst, count):

        if count >= len(lst):
            print("Count in sum_list with helper", count)
            return sum_so_far
        else:

            sum_list_helper(sum_so_far + lst[count], lst, count + 1)

    return sum_list_helper(0, x, 0)


s = list(range(6))

a = [1, 2, 3]
sum_list1 = SumList(s)
print("a:", a, SumList(a).sum())
for val in sum_list1:
    print(val)
print(sum_list1.sum())
# print(sum_right(s))

print(until(10, lambda x: not x % 3 or not x % 5, 0))
print(sum([n for n in range(10) if not n % 3 or not n % 5]))
print(sum_list(s))
