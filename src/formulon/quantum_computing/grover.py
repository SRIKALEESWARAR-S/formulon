import pennylane as qml
from dataclasses import dataclass
from typing import Tuple

@dataclass
class GroverResult:
    """Structure to hold the results of a Grover search."""
    found_state: int
    probabilities: Tuple[float, ...]


def _oracle(target: int, num_qubits: int) -> None:
    """Phase-flip target state |target>."""
    bits = f"{target:0{num_qubits}b}"

    for wire, bit in enumerate(bits):
        if bit == "0":
            qml.PauliX(wires=wire)

    target_wire = num_qubits - 1
    control_wires = list(range(num_qubits - 1))

    if control_wires:
        qml.ctrl(qml.PauliZ(wires=target_wire), control=control_wires)
    else:
        qml.PauliZ(wires=target_wire)

    for wire, bit in enumerate(bits):
        if bit == "0":
            qml.PauliX(wires=wire)


def _diffusion(num_qubits: int) -> None:
    """Apply the Grover diffusion operator (2|s><s| - I)."""
    for wire in range(num_qubits):
        qml.Hadamard(wires=wire)
        qml.PauliX(wires=wire)

    target_wire = num_qubits - 1
    control_wires = list(range(num_qubits - 1))

    if control_wires:
        qml.ctrl(qml.PauliZ(wires=target_wire), control=control_wires)
    else:
        qml.PauliZ(wires=target_wire)

    for wire in range(num_qubits):
        qml.PauliX(wires=wire)
        qml.Hadamard(wires=wire)


def grover_search(target: int, num_qubits: int = 2, iterations: int = 1) -> GroverResult:
    """Execute Grover's search algorithm and return the identified state and probabilities."""
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit():
        for wire in range(num_qubits):
            qml.Hadamard(wires=wire)

        for _ in range(iterations):
            _oracle(target, num_qubits)
            _diffusion(num_qubits)

        return qml.probs(wires=range(num_qubits))

    # Get probabilities from circuit
    probs = tuple(float(x) for x in circuit())
    
    # Identify the state with the highest probability
    found_state = max(range(len(probs)), key=lambda i: probs[i])
    
    return GroverResult(found_state=found_state, probabilities=probs)
