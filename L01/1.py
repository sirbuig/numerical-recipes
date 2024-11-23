# Ex 1

# numerical stability
def func(x, k):
    for _ in range(k):
        # print("%.4f" % x)
        print(x)
        if x < 0.5:
            x = 2 * x
        elif 0.5 < x <= 1:
            x = 2 * x - 1
    return x


print(func(0.1, 60))
