import asyncio
from functools import wraps
import sys

sys.setrecursionlimit(1000000000)
n = 0


def async_memoize(func):
    global n
    n = n + 1
    # The cache should be accessible within the inner function due to closures
    cache = {}
    # A dictionary to hold futures for results currently being computed,
    # preventing multiple calls for the same key at the same time.
    pending = {}

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Create a cache key from the function arguments
        # Note: args and kwargs should be hashable for the cache to work
        cache_key = args + tuple(sorted(kwargs.items()))

        # If the result is already in the cache, return it immediately
        if cache_key in cache:
            return cache[cache_key]

        # If a task is already running for this key, await its result
        if cache_key in pending:
            x = await pending[cache_key]
            print(x)
            return x

        # If not in cache and not pending, create a task (future) for the result
        loop = asyncio.get_running_loop()
        future = loop.create_future()  # ohlk
        pending[cache_key] = future

        try:
            # Call the original async function and await its result
            result = await func(*args, **kwargs)
            # Store the result in the cache
            cache[cache_key] = result
            # Set the result on the future for other pending calls
            future.set_result(result)
            return result
        except Exception as e:
            # If an error occurred, set the exception on the future
            future.set_exception(e)
            raise
        finally:
            # Once complete (either success or error), remove from pending
            if cache_key in pending:
                del pending[cache_key]
            # future.print_stack(limit=100)

    print(f"n={n}")
    return wrapper


def async_memoize2(func):

    cache = {}

    # @wraps(func)
    async def inner(*args, **kwargs):
        cache_key = (func.__name__, args, frozenset(kwargs))
        if cache_key in cache:
            return await cache[cache_key]

        task = asyncio.create_task(func(*args, **kwargs))
        cache[cache_key] = task

        try:
            result = await task
            # print(result)
            return result
        except Exception as e:
            del cache[cache_key]
            print(e)

    return inner


# Example async recursive function with the decorator
@async_memoize
async def async_fib(n):
    if n <= 1:
        return n
    # The recursive calls must also use 'await'
    res1 = await async_fib(n - 1)
    res2 = await async_fib(n - 2)
    return res1 + res2


# Example usage:
async def main():
    # Calling the function multiple times with the same input demonstrates memoization
    print(f"fib(10): {await async_fib(10)}")
    print(f"fib(500): {await async_fib(5000)}")
    print(f"fib(10) again: {await async_fib(100)}")  # This will be retrieved from cache


if __name__ == "__main__":
    asyncio.run(main())
    print(n)
