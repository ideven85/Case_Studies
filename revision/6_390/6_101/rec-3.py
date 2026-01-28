# 6.101 recitation: linked lists


# Python list vs. linked-list
lst1 = [1, 2, 3]
ll1 = (1, (2, (3, None)))

lst2 = [3]
ll2 = (3, None)

lst3 = []
ll3 = None


def make_ll(*elements):
    """
    given an arbitrary number of elements as arguments,
    make a linked-list of (first,rest) pairs

    >>> make_ll(1, 2, 3)
    (1, (2, (3, None)))
    """
    if not elements:
        return None
    return (elements[0], make_ll(elements[1:]))


def first(ll):
    """
    returns the first element of a non-empty linked list

    >>> first((5, (10, (15, None))))
    5
    """
    pass


def rest(ll):
    """
    returns the rest of a nonempty linked list (omitting the first element)

    >>> rest((5, (10, (15, None))))
    (10, (15, None))
    """
    pass


def ll_get(ll, ix):
    """
    get the ith element of a linked list

    >>> ll_get(('a',('b',None)), 1)
    'b'
    """
    pass


def ll_len(ll):
    """
    get the length of a linked list

    >>> ll_len(('a',('b',None)))
    2
    """
    pass


def ll_replace(ll, ix, val):
    """
    return a new linked list with the value at index ix replaced by the given
    value
    >>> ll_replace(make_ll(1, 2, 3), 1, 7)
    (1, (7, (3, None)))
    """
    pass


def ll_elements(ll):
    """
    generator that yields the elements in a linked list, so that it can be used
    in an iterative for loop like `for elt in ll_elements(ll)`

    >>> for i in ll_elements(make_ll(1, 2, 3)):
    ...     print(i)
    1
    2
    3
    """
    pass


if __name__ == "__main__":
    import doctest

    doctest.testmod()
