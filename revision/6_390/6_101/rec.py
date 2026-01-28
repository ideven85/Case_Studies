# 6.101 recitation: sets and dictionaries

# list, tuple, set, frozenset, and dict


# measuring performance


def make_big_list():
    return list(range(100_000_000))


def make_big_set():
    return set(range(100_000_000))


big_list = make_big_list()
big_set = make_big_set()


def look_in_big_list():
    return -1 in big_list


def look_in_big_set():
    return -1 in big_set


# choosing a data structure


def first_occurrence(data):
    """
    Given a list of integers or strings, return a new list with the same items
    in the same order, but keeping only the first occurrence of each item.
    """
    pass


def test_first_occurrence_small():
    assert first_occurrence([1, 9, 1, 1, 5, 3, 2, 9, 10]) == [1, 9, 5, 3, 2, 10]

    expected = ["hello", "world", "goodbye"]
    assert first_occurrence(["hello", "world", "goodbye", "hello"]) == expected


def test_first_occurrence_big_range():
    big = list(range(10_000_000))
    assert first_occurrence(big) == big
