"""
Question 1: Write the body of the tree_max function below.

You may test this function by running
`pytest tree_recursion.py -v`
without quotes.

For an extra challenge, try to write the function
using list comprehensions.
"""

from .debug_recursion import show_recursive_structure


@show_recursive_structure
def tree_max(tree):
    """
    Given tree as a dict { 'value': number,
                           'children': list of trees },
    return the maximum value found in the tree.
    """
    if len(tree) == 0:
        return 0
    key = tree["value"]
    values = tree["children"]
    # print(values)
    if len(values) == 0:
        return key
    else:
        return max(key, max(tree_max(x) for x in values))


# @show_recursive_structure
def tree_sum(tree):
    """
    Given tree as a dict { 'value': number,
                           'children': list of trees },
    return the sum of all the values found in the tree.
    """
    if len(tree) == 0:
        return 0
    val = tree["value"]
    children = tree["children"]
    if len(children) == 0:
        return val
    elif type(children) is list and len(children):
        print(children)
        return val + sum(tree_sum(x) for x in children)


count = 0


def flatten(lst):
    if not lst:
        return []
    if isinstance(lst[0], list):
        yield from flatten(lst[0])
    yield lst[1:]


@show_recursive_structure
def tree_list(tree):
    """
    Given tree as a dict { 'value': number,
                           'children': list of trees },
    return a sorted list of all the values found in the tree (from
    smallest to largest) sorted only once.
    """
    global count
    if not tree:
        return []
    val = tree["value"]
    children = tree["children"]
    count += 1
    if not children:
        return val
    if children:
        return [val] + [tree_list(x) for x in children]


def tree_list_v2(tree):
    """
    Returns a flat list of all values using a recursive one-liner.
    """
    if not tree:
        return []

    # [tree['value']] + sum([list_of_child_values_1],
    #                       [list_of_child_values_2]],
    #                      [])
    # The sum(..., []) is a "trick" to flatten a list of lists.
    return [tree["value"]] + sum([tree_list(child) for child in tree["children"]], [])


def tree_list_iterative(tree):
    if not tree:
        return []
    out = []
    agenda = [tree]
    while agenda:
        x = agenda.pop(-1)
        print(x)
        if not x:
            continue
        else:
            out.append(x["value"])
            if x["children"]:
                agenda.append(x["children"])
    return out


@show_recursive_structure
def tree_depth(tree, depth):
    """
    Given tree as a dict { 'value': number,
                           'children': list of trees },
    return a sorted list of all the values found in the tree (from
    smallest to largest) at the given depth (where depth=0 represents
    the top value, depth=1 represents the child values, etc.)

    If there are no values at the given depth, return None.
    """


t1 = {"value": 3, "children": []}

t2 = {
    "value": 9,
    "children": [{"value": 2, "children": []}, {"value": 11, "children": []}],
}

t3 = {
    "value": 9,
    "children": [
        {"value": 2, "children": []},
        {
            "value": 3,
            "children": [
                {"value": 99, "children": []},
                {"value": 16, "children": [{"value": 7, "children": []}]},
                {"value": 42, "children": []},
            ],
        },
    ],
}


def test_tree_max():
    assert tree_max(t1) == 3
    # assert tree_max(t2) == 11
    assert tree_max(t3) == 99
    print("correct!")


# def test_tree_sum():
#     assert tree_sum(t1) == 3
#     assert tree_sum(t2) == 21
#     assert tree_sum(t3) == 178
#     print("correct!")


#
#
#
#
def test_tree_list():
    assert tree_list(t1) == [3]
    assert tree_list(t2) == [2, 9, 11]
    assert tree_list(t3) == [2, 3, 7, 9, 16, 42, 99]
    print("correct!")


#
#
# def test_tree_depth():
#     assert tree_depth(t1, -1) is None
#     assert tree_depth(t1, 0) ==  [3]
#     assert tree_depth(t1, 1) is None
#     assert tree_depth(t2, 0) == [9]
#     assert tree_depth(t2, 1) == [2, 3, 7]
#     assert tree_depth(t2, 199) is None
#     assert tree_depth(t3, 0) == [9]
#     assert tree_depth(t3, 1) == [2, 3]
#     assert tree_depth(t3, 2) == [16, 42, 99]
#     assert tree_depth(t3, 3) == [7]
#     print("correct!")
#


if __name__ == "__main__":
    # uncomment the @show_recursive_structure line
    # above the function to see some detailed function output
    print(tree_list_iterative(t2))
