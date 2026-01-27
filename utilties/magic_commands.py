from IPython.core.magic import register_line_magic, magics_class
import time
from IPython.core.magic import register_cell_magic

# from utilties.clock_utils import clock_deco


from IPython.core.magic import Magics, line_magic, cell_magic
from IPython.terminal.pt_inputhooks import register


def load_ipython_extension(ipython):
    # If using function-based magics, they are registered by their decorators
    # If using class-based magics, instantiate and register them
    ipython.register(MyCustomMagics)  # Uncomment if using class-based magics
    ipython.register(clockit, "cell")
    ipython.register(my_cell_magic, "cell")  # todo check


@magics_class
class MyCustomMagics(Magics):
    @line_magic
    def my_class_line_magic(self, line):
        print(f"Class-based line magic: {line}")

    @cell_magic
    def my_class_cell_magic(self, line, cell):
        print(f"Class-based cell magic (line args: {line}):\n{cell}")


@register_line_magic
def line_printed(line):
    """A simple line magic that prints its input."""
    exp = line
    print(f"Entered {exp}")
    result = eval(line)
    print(f"{line}={result},{exp}")


# todo
@register_line_magic
def clockit(line):
    def clocked(*args):
        start = time.perf_counter_ns()
        result = line(*args)
        func_name = line.__name__
        end = time.perf_counter_ns() - start
        name = ",".join(repr(arg) for arg in args)
        print(f"Elapsed [{end:.8f}s],{func_name}({name})->{result}")
        return result

    return clocked


#
@register_cell_magic
def my_cell_magic(line, cell):
    """A simple cell magic that prints line arguments and cell content."""
    print(f"Line arguments: {line}")
    print(f"Cell 1 arguments: {cell}")


#
