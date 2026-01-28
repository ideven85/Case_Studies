"""
Question 1: Write the body of the tree_max function below.

You may test this function by running
`pytest q1_tree_max.py -v`
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
    value = tree["value"]
    children = tree["children"]
    if not children:
        return value

    else:

        return max(max(tree_max(x) for x in children), value)


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


def test_tree_max():
    assert tree_max(t1) == 3
    assert tree_max(t2) == 9
    assert tree_max(t3) == 99
    print("correct!")


if __name__ == "__main__":
    # uncomment the @show_recursive_structure line
    # above the function to see some detailed function output
    tree_max(t3)
