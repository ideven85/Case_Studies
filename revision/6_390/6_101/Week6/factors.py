def factors(n):
    i = 1
    # yield 1

    while i * i < n:
        if n % i == 0:
            yield i
            yield n // i

            # print(i,n//i)

        i += 1
    # yield n


print(sorted(list(factors(16))))
