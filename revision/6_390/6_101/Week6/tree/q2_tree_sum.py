"""
Question 2: Write the body of the tree_sum function below.

You may test this function by running
`pytest q2_tree_sum.py -v`
without quotes.
"""

# from debug_recursion import show_recursive_structure


def tree_sum_recursive(tree):
    if not tree:
        return 0
    if not tree["children"]:
        return tree["value"]
    return tree["value"] + sum(tree_sum_recursive(x) for x in tree["children"])


def tree_sum(tree):
    """
    Given tree as a dict { 'value': number,
                           'children': list of trees },
    return the sum of all the values found in the tree.
    """

    sum_so_far = 0
    if not tree:
        return sum_so_far
    # Convert tree to List

    agenda = [tree]
    while agenda:
        x = agenda.pop(-1)
        # print(x)
        if not x:
            sum_so_far += 0
        else:
            sum_so_far += x

    return sum_so_far


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


if __name__ == "__main__":
    # uncomment the @show_recursive_structure line
    # above the function to see some detailed function output
    print(tree_sum_recursive(t1))
    print(tree_sum_recursive(t2))
    print(tree_sum_recursive(t3))

    print("\n\n", tree_sum(t1), "\n", tree_sum(t2), "\n", tree_sum(t3))
