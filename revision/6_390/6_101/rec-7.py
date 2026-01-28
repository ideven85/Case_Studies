# 6.101 recitation 0
# subject introduction
# testing and debugging, some python "goodies" and style


############### EXAMPLE 0: ADDING ELEMENTS IN NESTED LISTS


def sum_lists(lists):
    """
    Given a list of lists, return a new list where each list is replaced by
    the sum of its elements.
    """
    output = [0] * len(lists)
    for i in range(len(lists)):
        total = 0
        for i in lists[i]:
            total += i
        output[i] = total
    return output


############### EXAMPLE 1: REVERSING NESTED LISTS


def reverse_all(inp):
    """
    given a list of lists, return a new list of lists
    but with all of the inner lists reversed, without
    modifying the input list

    example:
    >>> input1 = [[1, 2], [3, 4]]
    >>> reverse_all(input1)
    [[2, 1], [4, 3]]
    """
    output_copy = inp
    for L in output_copy:
        L.reverse()
    return output_copy


############### EXAMPLE 2: SUBTRACTING CORRESPONDING ELEMENTS IN LISTS


def subtract_lists(l1, l2):
    """
    Given lists of numbers l1 and l2, return a new list where each value is
    the difference between the corresponding values in l1 and in l2.
    """
    output = []
    for i in range(len(l1)):
        output.append(l1[i] - l2[i])
    return output
