from functools import wraps


def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Calling decorated function")
        return func(*args, **kwargs)

    return wrapper


@my_decorator
def my_function():
    print("Example Function")


my_function()
