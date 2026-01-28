def sum_nested_list(tree):
    if not tree:
        return 0
    elif isinstance(tree[0], list):
        return sum_nested_list(tree[0]) + sum_nested_list(tree[1:])
    else:
        return tree[0] + sum_nested_list(tree[1:])


def sum_nested(original_x):
    total = 0
    agenda = [original_x]
    count = 0
    while agenda:
        x = agenda.pop(-1)
        count += 1
        if not x:
            continue

        elif isinstance(x[0], list):
            agenda.append(x[0])
            agenda.append(x[1:])
        else:
            total += x[0]
            agenda.append(x[1:])
    print(count)
    return total


#
# t1 = {"value": 3, "children": []}
#
# t2 = {
#     "value": 9,
#     "children": [
#         {"value": 2, "children": []},
#         {"value": 3, "children": []},
#         {"value": 7, "children": []},
#     ],
# }
#
# t3 = {
#     "value": 9,
#     "children": [
#         {"value": 2, "children": []},
#         {
#             "value": 3,
#             "children": [
#                 {"value": 99, "children": []},
#                 {"value": 16, "children": [{"value": 7, "children": []}]},
#                 {"value": 42, "children": []},
#             ],
#         },
#     ],
# }
nested_tree1 = [3, []]
nested_tree2 = [[9, [[2, []], [3, []], [7, []]]]]
nested_tree3 = [9, [[2, []], [3, [99, []], [16, [7, []]], [42, []]]]]
print(sum_nested_list(nested_tree1))
print(sum_nested(nested_tree2))
print(sum_nested(nested_tree3))
