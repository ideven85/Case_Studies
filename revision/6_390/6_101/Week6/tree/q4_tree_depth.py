"""
Question 4: Write the body of the tree_depth function below.

You may test this function by running
`pytest q4_tree_depth.py -v`
without quotes.
"""

from debug_recursion import show_recursive_structure


# @show_recursive_structure
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
    "children": [
        {"value": 2, "children": []},
        {"value": 3, "children": []},
        {"value": 7, "children": []},
    ],
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


def test_tree_depth():
    assert tree_depth(t1, -1) is None
    assert tree_depth(t1, 0) == [3]
    assert tree_depth(t1, 1) is None
    assert tree_depth(t2, 0) == [9]
    assert tree_depth(t2, 1) == [2, 3, 7]
    assert tree_depth(t2, 199) is None
    assert tree_depth(t3, 0) == [9]
    assert tree_depth(t3, 1) == [2, 3]
    assert tree_depth(t3, 2) == [16, 42, 99]
    assert tree_depth(t3, 3) == [7]
    print("correct!")


if __name__ == "__main__":
    # uncomment the @show_recursive_structure line
    # above the function to see some detailed function output
    print(tree_depth(t3, 3))
