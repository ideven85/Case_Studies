"""
Question 3: Write the body of the tree_list function below.

You may test this function by running
`pytest q3_tree_list.py -v`
without quotes.

Is there a way you can write this function that requires
sorting the result only once?
"""

from .debug_recursion import show_recursive_structure

count = 0


# todo
# @show_recursive_structure
# def tree_list(tree):
#     """
#     Given tree as a dict { 'value': number,
#                            'children': list of trees },
#     return a sorted list of all the values found in the tree (from
#     smallest to largest) sorted only once.
#     """
#
#     # --- 1. Define an inner "helper" function for recursion ---
#     # This function will just gather all the values in a flat list
#     def _gather_values(subtree):
#         if not subtree:
#             return []
#
#         # Get the flat, unsorted list (pre-order)
#         return [subtree['value']] + sum(
#             [_gather_values(child) for child in subtree['children']], []
#         )
#
#     # --- 2. Call the helper function to get all values ---
#     flat_list = _gather_values(tree)
#
#     # --- 3. Sort the final list ONCE, just before returning ---
#     return sorted(flat_list)


def tree_list(tree):
    """
    Returns a FLAT, unsorted list of all values.
    """
    if not tree:
        return []

    # This sums all the child lists together into one flat list
    # and adds it to the current node's value list.
    return sorted(
        [tree["value"]] + sum([tree_list(child) for child in tree["children"]], [])
    )


def gen_tree_list(tree):
    if not tree:
        return []
    out = []
    value = tree["value"]
    children = tree["children"]
    out.append(value)
    if children:
        temp = (gen_tree_list(x) for x in children)
        out.extend(temp)
    return out


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
                {"value": 16, "children": []},
                {"value": 7, "children": []},
                {"value": 42, "children": []},
            ],
        },
    ],
}


def test_tree_list():
    assert tree_list(t1) == [3]
    assert tree_list(t2) == [2, 3, 7, 9]
    assert tree_list(t3) == [2, 3, 7, 9, 16, 42, 99]
    print("correct!")


if __name__ == "__main__":
    # uncomment the @show_recursive_structure line
    # above the function to see some detailed function output
    # print(tree_list(t1))
    print(tree_list(t2))
    # print(tree_list(t3))
    print(gen_tree_list(t2))
    # print(tree_list(t1))
