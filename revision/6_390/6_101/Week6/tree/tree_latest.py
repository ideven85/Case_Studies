def tree_max(tree):
    if not tree:
        return 0
    maximum = float("-inf")
    agenda = [tree]
    while agenda:
        current = agenda.pop(0)
        print(maximum)
        if not current:
            continue
        if isinstance(current, list):
            print("Length,", len(current), current[0], current)
            current = current[0]

            if current:
                maximum = max(current["value"], maximum)
                if current["children"]:
                    print("Hello")
                    agenda.append(current["children"])
        else:
            maximum = max(current["value"], maximum)
            if current["children"]:
                print("Hi", current)
                current = current["children"].pop()
                maximum = max(current["value"], maximum)

                agenda.append(current["children"])
    print(agenda)
    return maximum


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
                {"value": 9, "children": []},
                {"value": 160, "children": [{"value": 7, "children": []}]},
                {"value": 42, "children": []},
            ],
        },
    ],
}


def test_tree_max():
    # assert tree_max(t1) == 3
    # assert tree_max(t2) == 9
    assert tree_max(t3) == 160
    print("correct!")
