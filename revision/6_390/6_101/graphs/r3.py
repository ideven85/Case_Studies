def print_el(*args):

    print("\n".join([str(el)[1:-1] for el in zip(*args)]))


def main():
    x = [1, 2, 3, 4]
    y = [el / 2 for el in x]
    z = [el * 2 for el in x]
    for el in zip(x, y, z):
        print(*el)
    print_el(x, y, z)


if __name__ == "__main__":
    main()
