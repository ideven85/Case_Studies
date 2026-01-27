import pytest


class CustomException(Exception):
    def __init__(self, msg=None):
        if not msg:
            print("Oops something went wrong,No reason")

        else:
            print(msg)


class DateTimeFormatException(Exception):
    pass


def divide(numerator, denominator):
    if denominator == 0:
        with pytest.raises(ZeroDivisionError) as e:
            assert "This is being caught" in e.value
    else:
        return numerator / denominator


def divide_2(numerator, denominator):
    if denominator == 0:
        raise ZeroDivisionError("Cannot divide by zero!")
    return numerator / denominator


def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError) as excinfo:
        divide_2(10, 0)
    assert "Cannot divide by zero!" in str(excinfo.value)


def raise_custom_exception():

    raise CustomException("This is being caught")


# todo check pytest documentation for custom exceptions
def test_divide_by_zero_2():
    assert divide(10, 2) == (10 / 2)
    with pytest.raises(CustomException) as execinfo:
        raise_custom_exception()
    assert execinfo.value == "This is being caught"


#
# def test_exception():
#     with pytest.raises(CustomException) as execinfo:
#         raise raise_custom_exception()
#     assert str(execinfo.value) == "This is being caught"
