import time


def clock_deco(func):
    def clocked(*args):
        start = time.perf_counter_ns()
        result = func(*args)
        func_name = func.__name__
        end = time.perf_counter_ns() - start
        name = ",".join(repr(arg) for arg in args)
        print(f"Elapsed [{end:.8f}s],{func_name}({name})->{result}")
        return result

    return clocked
