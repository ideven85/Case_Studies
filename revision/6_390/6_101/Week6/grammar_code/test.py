from grammar import all_phrases

test_grammar_1 = {
    "sentence": [["noun", "verb"], ["noun", "never", "verb"]],
    "noun": [["pigs"], ["professors"]],
    "verb": [["fly"], ["think"]],
}

test_grammar_2 = {
    "start": [["n"], ["adj", "n"], ["adj", "adj", "n"]],
    "adj": [["quirky"], ["hungry"], ["n"]],
    "n": [["cat"], ["dog"]],
}

test_grammar_3 = {
    "greeting": [["hi", "there"], ["hi", "name"]],
    "name": [["Rosenkrantz"], ["Guildenstern"]],
}

test_grammar_4 = {"foo": [["End", "of", "recitation"]]}


def _verify_all_phrases_result(L):
    # top level is a list
    if not isinstance(L, set):
        return False
    # each phrase is a list of strings
    for phrase in L:
        if not isinstance(phrase, tuple):
            return False
        for terminal in phrase:
            if not isinstance(terminal, str):
                return False
    return True


def test_all_phrases_1():
    all_phrases(test_grammar_1, "pigs") == {("pigs",)}


def test_all_phrases_2():
    expected = {("pigs",), ("professors",)}
    result = all_phrases(test_grammar_1, "noun")
    assert result == expected, f"{result=} != {expected=}"


def test_all_phrases_3():
    result = all_phrases(test_grammar_1, "sentence")
    expected = {
        (
            "pigs",
            "fly",
        ),
        (
            "pigs",
            "think",
        ),
        ("professors", "fly"),
        ("professors", "think"),
        ("pigs", "never", "fly"),
        ("pigs", "never", "think"),
        ("professors", "never", "fly"),
        ("professors", "never", "think"),
    }
    assert result == expected


def test_all_phrases_4():
    expected = {("hungry",), ("quirky",), ("cat",), ("dog",)}
    assert all_phrases(test_grammar_2, "adj") == expected


def test_all_phrases_5():
    result = all_phrases(test_grammar_2, "start")
    expected = {
        ("cat",),
        ("dog",),
        ("hungry", "cat"),
        ("hungry", "dog"),
        ("quirky", "cat"),
        ("quirky", "dog"),
        ("cat", "cat"),
        ("cat", "dog"),
        ("dog", "cat"),
        ("dog", "dog"),
        ("hungry", "hungry", "cat"),
        ("hungry", "hungry", "dog"),
        ("hungry", "quirky", "cat"),
        ("hungry", "quirky", "dog"),
        ("hungry", "cat", "cat"),
        ("hungry", "cat", "dog"),
        ("hungry", "dog", "cat"),
        ("hungry", "dog", "dog"),
        ("quirky", "hungry", "cat"),
        ("quirky", "hungry", "dog"),
        ("quirky", "quirky", "cat"),
        ("quirky", "quirky", "dog"),
        ("quirky", "cat", "cat"),
        ("quirky", "cat", "dog"),
        ("quirky", "dog", "cat"),
        ("quirky", "dog", "dog"),
        ("cat", "hungry", "cat"),
        ("cat", "hungry", "dog"),
        ("cat", "quirky", "cat"),
        ("cat", "quirky", "dog"),
        ("cat", "cat", "cat"),
        ("cat", "cat", "dog"),
        ("cat", "dog", "cat"),
        ("cat", "dog", "dog"),
        ("dog", "hungry", "cat"),
        ("dog", "hungry", "dog"),
        ("dog", "quirky", "cat"),
        ("dog", "quirky", "dog"),
        ("dog", "cat", "cat"),
        ("dog", "cat", "dog"),
        ("dog", "dog", "cat"),
        ("dog", "dog", "dog"),
    }
    assert result == expected


def test_all_phrases_6():
    result = all_phrases(test_grammar_3, "greeting")
    expected = {("hi", "there"), ("hi", "Rosenkrantz"), ("hi", "Guildenstern")}
    assert result == expected


def test_all_phrases_7():
    result = all_phrases(test_grammar_4, "foo")
    expected = {("End", "of", "recitation")}
    assert result == expected
