from typing import TypeVar, Callable
from contextlib import AbstractContextManager, contextmanager

T = TypeVar("T")
V = TypeVar("V")


class Gfg:
    def __init__(self):
        self.count = 0
        print("Instance Created")

    # Defining __call__ method
    def __call__(self):
        self.count += 1
        print("Instance is called via special method")
        return self.count

    def __enter__(self):
        print("Enter")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exit")

    def __str__(self):
        return "GFG"


class Open:
    def __init__(self, file, mode):
        self.count = 0
        self.file = file
        self.mode = mode
        self.name = None

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.count

    def __enter__(self):
        self.name = open(self.file, self.mode)

        return self.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.name.close()


def _with(ctx: AbstractContextManager[V], action: Callable[[T], V]):
    with ctx as c:
        return action(c)


def main():
    a = Gfg()
    with a as manager:
        print(manager)

    x = Open("test.txt", "r")

    with x as f:
        print(f.read())
    print("\n", a())
    ctx_manager = open("key.txt", "r")
    action = lambda: _with(ctx_manager, lambda f: f)
    print(action())


main()
