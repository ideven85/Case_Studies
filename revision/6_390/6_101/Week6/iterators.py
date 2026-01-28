"""
1) 1) What is an iterable? An iterator? A generator? What are the similarities and
differences between them?
"""

# Doubt exactly what a generator object is because java does not have generators, but generators
# are used to stopping iteration and giving current element, like say in Java, IntStream.rangeClosed(1,100).parallel.filter(x->x%2!=0).peek()
# Peek is an intermediary operation not a terminal operation in streams in Java, in Python I cannot describe exactly what it means

"""
Iterable is an object which we can iterate over
Iterator is an object which iterates over an iterable
Generators are objects which yield the current element # Not clear Need practice
"""

"""
2) How do you make an iterator "from an iterable object in Python?"
Calling next with a generator passed in will cause Python to start running the body of the code, 
run until we see a yield, pause the execution again, and give us the value that was yielded from the generator.
Object is already iterable... either by implementing __iter__ or "__getitem_- which fetches bu in index"
 
"""


# class LinkedStack:
#     class Node:
#         slots = '_element','_next'
#         def __init__(self,value):
#             self._element = value
#             self._next = None
#


class StackList:
    def __init__(self):
        self._stack = []
        self._length = 0

    def push(self, item):
        n = 1
        if type(item) in (str, int, float):
            self._stack.append(item)
        elif type(item) in (list, set, tuple):
            self._stack.extend(item)
            n = len(item)
        self._length += n

    def pop(self):

        if len(self) > 0:
            self._length -= 1
            return self._stack[self._length]
        else:
            raise EmptyException("Stack Is Empty")
            # print("Stack is empty, please add some more elements before pooping", e)

    def __iadd__(self, item):
        self.push(item)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if len(self):
            return self._stack[index]
        else:
            raise EmptyException("Stack is Empty")

    def __iter__(self):
        yield self

    def __next__(self):
        if self._length == 0:
            raise StopIteration("Stack is Empty")
        self._length -= 1
        return self._stack[self._length]

    def __repr__(self):
        return ",".join(str(x) for x in self._stack)

    # def __str__(self):
    #     return self._stack


class EmptyException(Exception):
    pass


if __name__ == "__main__":
    obj = StackList()
    obj.push(1)
    obj.push(2)
    obj.push(3)
    obj.push(4)
    obj.push((5, 6))
    # obj+=[2]
    # print(obj)
    #
    # for i in range(len(obj)):
    #     print(obj[i], end=" ")
    print("Obj:", obj)
    for val in obj:
        print(val, end=",")
    print()
    # for val in obj:
    #     print(val, end=" ")
    # print("\nIteration over")
    print(obj.pop())
    print(obj.pop())
    print(obj.pop())
    print(obj.pop())
    # print(obj.pop())  # Error
