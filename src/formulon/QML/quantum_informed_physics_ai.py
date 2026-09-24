"""
Formulon v1.0
Quantum-Informed Physical AI Engine
===================================

Pipeline
--------
Random physical signal
    -> physics feature generation
    -> Qiskit noisy quantum channel
    -> noisy quantum measurements
    -> PennyLane trainable quantum layer
    -> PyTorch ML decoder
    -> reconstructed clean physical features
    -> physics-pattern classification

The design intentionally keeps the quantum part small enough to run on a
modest computer. It is a research prototype, not a claim of quantum
advantage.

Scientific note:
The Qiskit noisy channel is a classical simulation of a specified error
model. The PennyLane circuit is also classically simulable at this scale.
The purpose is to study whether a hybrid quantum-informed representation
can learn useful pre-/post-noise physical mappings, not to claim quantum
advantage.

Install:
    pip install numpy scipy scikit-learn matplotlib torch pennylane qiskit qiskit-aer

Run:
    python quantum_informed_physics_ai.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from scipy import special
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Optional quantum dependencies
# -----------------------------
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, phase_damping_error


# ============================================================
# Configuration
# ============================================================

SEED = 42

N_SAMPLES = 300
N_QUBITS = 8
N_SHOTS = 256

SIGNAL_LENGTH = 64
FEATURE_DIM = 8

TEST_SIZE = 0.20
EPOCHS = 60
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

QISKIT_ERROR_PROB = 0.025
PHASE_DAMPING = 0.015



def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Layer 1 — Highly Random Physical Signal Generator
# ============================================================

PATTERNS = [
    "harmonic",
    "damped",
    "gaussian_wave",
    "quadratic",
]


def generate_random_physical_signal(
    pattern: str,
    n: int = SIGNAL_LENGTH,
) -> np.ndarray:
    """
    Generate a randomized synthetic physical signal.

    The randomization is deliberate: amplitude, frequency, phase,
    damping, offset and noise vary between samples.
    """
    t = np.linspace(-1.0, 1.0, n)

    amplitude = np.random.uniform(0.4, 1.4)
    frequency = np.random.uniform(1.0, 5.0)
    phase = np.random.uniform(-math.pi, math.pi)
    offset = np.random.uniform(-0.15, 0.15)

    if pattern == "harmonic":
        y = amplitude * np.sin(2 * math.pi * frequency * t + phase)

    elif pattern == "damped":
        damping = np.random.uniform(0.2, 1.4)
        y = (
            amplitude
            * np.exp(-damping * (t + 1.0))
            * np.sin(2 * math.pi * frequency * t + phase)
        )

    elif pattern == "gaussian_wave":
        center = np.random.uniform(-0.35, 0.35)
        width = np.random.uniform(0.12, 0.35)
        carrier = np.sin(2 * math.pi * frequency * t + phase)
        envelope = np.exp(-((t - center) ** 2) / (2 * width**2))
        y = amplitude * envelope * carrier

    elif pattern == "quadratic":
        a = np.random.uniform(0.5, 2.0) * np.random.choice([-1.0, 1.0])
        b = np.random.uniform(-1.0, 1.0)
        y = a * t**2 + b * t + offset

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    noise = np.random.normal(0.0, np.random.uniform(0.005, 0.035), n)
    y = y + noise + offset

    # Stable normalization before feature extraction.
    y = y - np.mean(y)
    y = y / (np.std(y) + 1e-8)

    return y.astype(np.float64)


# ============================================================
# Layer 2 — Physics Feature Extraction
# ============================================================

def extract_physics_features(signal: np.ndarray) -> np.ndarray:
    """
    Convert a physical signal into a compact physics-informed vector.

    Features:
        0-3 : cubic Legendre polynomial coefficients
        4   : Bessel-shaped mean
        5   : Bessel-shaped standard deviation
        6   : first-difference energy
        7   : spectral centroid
    """
    x = np.linspace(-1.0, 1.0, len(signal))

    # Legendre polynomial basis.
    legendre = np.polynomial.legendre.legfit(x, signal, deg=3)

    # Bessel transformation.
    bessel = signal * special.jv(0, np.abs(signal) * 2.5)

    # Local dynamics.
    diff_energy = np.mean(np.diff(signal) ** 2)

    # Spectral centroid.
    spectrum = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(len(signal))
    spectral_centroid = float(
        np.sum(frequencies * spectrum) / (np.sum(spectrum) + 1e-8)
    )

    features = np.array(
        [
            legendre[0],
            legendre[1],
            legendre[2],
            legendre[3],
            np.mean(bessel),
            np.std(bessel),
            diff_energy,
            spectral_centroid,
        ],
        dtype=np.float64,
    )

    return features


# ============================================================
# Layer 3 — Qiskit Noisy Quantum Channel
# ============================================================

@dataclass
class QiskitQuantumResult:
    noisy_features: np.ndarray
    counts: Dict[str, int]


def build_noise_model() -> NoiseModel:
    """
    Build an intentionally imperfect quantum channel.

    This is where the synthetic 'hardware error' is injected.
    """
    noise_model = NoiseModel()

    one_qubit_depolarizing = depolarizing_error(
        QISKIT_ERROR_PROB,
        1,
    )

    two_qubit_depolarizing = depolarizing_error(
        QISKIT_ERROR_PROB,
        2,
    )

    phase_noise = phase_damping_error(PHASE_DAMPING)

    noise_model.add_all_qubit_quantum_error(
        one_qubit_depolarizing,
        ["rx", "ry", "rz"],
    )

    noise_model.add_all_qubit_quantum_error(
        phase_noise,
        ["rx", "ry", "rz"],
    )

    noise_model.add_all_qubit_quantum_error(
        two_qubit_depolarizing,
        ["cx"],
    )

    return noise_model


QISKIT_NOISE_MODEL = build_noise_model()
QISKIT_SIMULATOR = AerSimulator(
    noise_model=QISKIT_NOISE_MODEL,
    seed_simulator=SEED,
)


def features_to_angles(features: np.ndarray) -> np.ndarray:
    """
    Map every physical feature to its own rotation angle.

    The previous prototype averaged pairs of features into four angles,
    which destroyed information before the quantum circuit saw it.
    Formulon v1.0 instead uses one angle per physical feature.
    """
    features = np.asarray(features, dtype=np.float64)

    if features.size != FEATURE_DIM:
        raise ValueError(
            f"Expected {FEATURE_DIM} physical features, got {features.size}"
        )

    return np.pi * np.tanh(features)


def qiskit_noisy_channel(
    features: np.ndarray,
    shots: int = N_SHOTS,
) -> QiskitQuantumResult:
    """
    Encode physics features into a Qiskit circuit, deliberately apply
    a noisy channel, and recover measurement probabilities.

    The returned vector has one probability per computational basis state.
    """
    angles = features_to_angles(features)

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)

    # Data encoding.
    for q, angle in enumerate(angles):
        qc.ry(float(angle), q)
        qc.rz(float(angle / 2.0), q)

    # Entangling physical-information layer.
    for q in range(N_QUBITS - 1):
        qc.cx(q, q + 1)

    qc.barrier()

    qc.measure(range(N_QUBITS), range(N_QUBITS))

    job = QISKIT_SIMULATOR.run(
        qc,
        shots=shots,
        seed_simulator=SEED + int(np.sum(np.abs(features)) * 1_000_000) % 1_000_000,
    )
    result = job.result()
    counts = result.get_counts()

    dim = 2 ** N_QUBITS
    noisy_probabilities = np.zeros(dim, dtype=np.float64)

    for bitstring, count in counts.items():
        index = int(bitstring, 2)
        noisy_probabilities[index] = count / shots

    return QiskitQuantumResult(
        noisy_features=noisy_probabilities,
        counts=counts,
    )


# ============================================================
# Layer 4 — PennyLane Quantum Decoder
# ============================================================

pl_dev = qml.device(
    "default.qubit",
    wires=N_QUBITS,
    shots=None,
)


def probability_vector_to_angles(
    probability_vector: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a normalized 2**N_QUBITS probability vector to N_QUBITS
    trainable-scale inputs.

    The original implementation used chunk means, producing angles close
    to zero. Here we use a fixed logit-like transform followed by chunk
    aggregation so the quantum circuit sees a useful dynamic range.
    """
    p = probability_vector / (torch.sum(probability_vector) + 1e-8)

    # Center and amplify deviations from the uniform distribution.
    centered = (p - (1.0 / p.numel())) * p.numel()
    chunks = torch.chunk(centered, N_QUBITS)

    features = torch.stack(
        [torch.mean(chunk) for chunk in chunks]
    )

    return torch.pi * torch.tanh(2.5 * features)


@qml.qnode(pl_dev, interface="torch")
def pennylane_decoder_qnode(
    x: torch.Tensor,
    weights: torch.Tensor,
):
    """
    Trainable quantum feature map.

    Qiskit provides the noisy channel.
    PennyLane provides the differentiable quantum layer.
    """
    for q in range(N_QUBITS):
        qml.RY(x[q], wires=q)
        qml.RZ(x[q] / 2.0, wires=q)

    for q in range(N_QUBITS - 1):
        qml.CNOT(wires=[q, q + 1])

    for layer in range(weights.shape[0]):
        for q in range(N_QUBITS):
            qml.RY(weights[layer, q, 0], wires=q)
            qml.RZ(weights[layer, q, 1], wires=q)

        for q in range(N_QUBITS - 1):
            qml.CNOT(wires=[q, q + 1])

    return tuple(qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS))


class PennyLaneQuantumLayer(nn.Module):
    def __init__(self, n_layers: int = 2):
        super().__init__()

        self.weights = nn.Parameter(
            0.05 * torch.randn(n_layers, N_QUBITS, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []

        for sample in x:
            angles = probability_vector_to_angles(sample)

            result = pennylane_decoder_qnode(
                angles,
                self.weights,
            )

            outputs.append(torch.stack(result))

        return torch.stack(outputs)


# ============================================================
# Layer 5 — Quantum-Informed Physical AI Decoder
# ============================================================

class QuantumInformedPhysicsAI(nn.Module):
    """
    Learns the relationship:

        clean physical features
              ↓
        quantum encoding
              ↓
        noisy quantum channel
              ↓
        PennyLane quantum feature map
              ↓
        classical ML decoder
              ↓
        reconstructed physical features
    """

    def __init__(self):
        super().__init__()

        self.quantum_layer = PennyLaneQuantumLayer(n_layers=2)

        self.decoder = nn.Sequential(
            nn.Linear(N_QUBITS, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, FEATURE_DIM),
        )

        # Classification has a parallel path so it does not depend entirely
        # on the reconstruction bottleneck.
        self.classifier = nn.Sequential(
            nn.Linear(N_QUBITS + FEATURE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, len(PATTERNS)),
        )

    def forward(self, noisy_quantum_features: torch.Tensor):
        quantum_features = self.quantum_layer(
            noisy_quantum_features
        )

        reconstructed_features = self.decoder(
            quantum_features
        )

        classifier_input = torch.cat(
            [quantum_features, reconstructed_features],
            dim=1,
        )

        logits = self.classifier(classifier_input)

        return reconstructed_features, logits


# ============================================================
# Layer 6 — Dataset Construction
# ============================================================

@dataclass
class DatasetBundle:
    physics_features: np.ndarray
    noisy_quantum_features: np.ndarray
    labels: np.ndarray
    raw_signals: List[np.ndarray]


def build_dataset(n_samples: int = N_SAMPLES) -> DatasetBundle:
    physics_features = []
    noisy_features = []
    labels = []
    raw_signals = []

    print("\n[1/4] Generating highly randomized physical signals...")

    for i in range(n_samples):
        label = np.random.randint(len(PATTERNS))
        pattern = PATTERNS[label]

        signal = generate_random_physical_signal(pattern)
        features = extract_physics_features(signal)

        quantum_result = qiskit_noisy_channel(features)

        physics_features.append(features)
        noisy_features.append(quantum_result.noisy_features)
        labels.append(label)
        raw_signals.append(signal)

        if (i + 1) % 25 == 0:
            print(f"    Generated {i + 1}/{n_samples}")

    return DatasetBundle(
        physics_features=np.asarray(physics_features),
        noisy_quantum_features=np.asarray(noisy_features),
        labels=np.asarray(labels),
        raw_signals=raw_signals,
    )


# ============================================================
# Layer 7 — Training
# ============================================================

def train_model(
    model: QuantumInformedPhysicsAI,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    labels_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    labels_val: torch.Tensor,
) -> None:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    reconstruction_loss = nn.MSELoss()
    classification_loss = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None
    patience = 10
    stale = 0

    print("\n[3/4] Training Quantum-Informed Physical AI...")

    n = len(x_train)

    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(n)
        total_loss = 0.0

        for start_idx in range(0, n, BATCH_SIZE):
            idx = permutation[start_idx:start_idx + BATCH_SIZE]

            xb = x_train[idx]
            yb = y_train[idx]
            lb = labels_train[idx]

            optimizer.zero_grad()

            reconstructed, logits = model(xb)

            loss_reconstruction = reconstruction_loss(
                reconstructed,
                yb,
            )

            loss_classification = classification_loss(
                logits,
                lb,
            )

            loss = loss_reconstruction + 0.25 * loss_classification

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)

        # Validation checkpoint.
        model.eval()
        with torch.no_grad():
            val_reconstructed, val_logits = model(x_val)
            val_loss = (
                reconstruction_loss(val_reconstructed, y_val)
                + 0.25 * classification_loss(val_logits, labels_val)
            ).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / n
            print(
                f"    Epoch {epoch + 1:03d}/{EPOCHS} "
                f"| Train = {avg_loss:.6f} "
                f"| Val = {val_loss:.6f}"
            )

        if stale >= patience:
            print(f"    Early stopping at epoch {epoch + 1}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"    Restored best validation loss: {best_val:.6f}")


# ============================================================
# Layer 8 — Evaluation
# ============================================================

def evaluate_model(
    model: QuantumInformedPhysicsAI,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    labels_test: torch.Tensor,
    scaler: StandardScaler,
) -> None:

    model.eval()

    with torch.no_grad():
        reconstructed, logits = model(x_test)

    reconstructed_np = reconstructed.numpy()
    target_np = y_test.numpy()

    pred = torch.argmax(logits, dim=1).numpy()
    true = labels_test.numpy()

    mse = mean_squared_error(
        target_np,
        reconstructed_np,
    )

    accuracy = accuracy_score(
        true,
        pred,
    )

    # Convert the standardized reconstruction back to the physical
    # feature scale for a more interpretable error.
    target_original = scaler.inverse_transform(target_np)
    reconstructed_original = scaler.inverse_transform(
        reconstructed_np
    )

    original_mse = mean_squared_error(
        target_original,
        reconstructed_original,
    )

    print("\n[4/4] Evaluation")
    print("=" * 60)
    print(f"Feature reconstruction MSE (scaled): {mse:.6f}")
    print(f"Feature reconstruction MSE (physical): {original_mse:.6f}")
    print(f"Physics-pattern classification accuracy: {accuracy:.2%}")

    print("\nPattern results:")
    for true_label, pred_label in zip(true[:20], pred[:20]):
        print(
            f"    True: {PATTERNS[true_label]:15s} "
            f"| Predicted: {PATTERNS[pred_label]}"
        )


# ============================================================
# Layer 9 — Demonstration of Error Recovery
# ============================================================

def demonstrate_single_recovery(
    model: QuantumInformedPhysicsAI,
    sample_signal: np.ndarray,
    scaler: StandardScaler,
) -> None:

    true_features = extract_physics_features(sample_signal)

    quantum_result = qiskit_noisy_channel(
        true_features,
        shots=N_SHOTS,
    )

    noisy = quantum_result.noisy_features

    noisy_tensor = torch.tensor(
        noisy,
        dtype=torch.float32,
    ).unsqueeze(0)

    model.eval()

    with torch.no_grad():
        reconstructed, logits = model(noisy_tensor)

    reconstructed_scaled = reconstructed.numpy()
    reconstructed_features = scaler.inverse_transform(
        reconstructed_scaled
    )[0]

    predicted_class = int(
        torch.argmax(logits, dim=1).item()
    )

    print("\n" + "=" * 60)
    print("SINGLE QUANTUM ERROR-RECOVERY DEMONSTRATION")
    print("=" * 60)

    print("\nOriginal physical features:")
    print(np.round(true_features, 5))

    print("\nNoisy quantum measurement:")
    print(np.round(noisy, 5))

    print("\nML-reconstructed physical features:")
    print(np.round(reconstructed_features, 5))

    print(
        f"\nPredicted physical pattern: "
        f"{PATTERNS[predicted_class]}"
    )

    print("\nQuantum measurement counts:")
    print(quantum_result.counts)


# ============================================================
# Layer 10 — Main
# ============================================================

def main():
    seed_everything()

    print("=" * 60)
    print("FORMULON v1.0")
    print("QUANTUM-INFORMED PHYSICAL AI ENGINE")
    print("=" * 60)

    print("\nArchitecture:")
    print("  Random Physics")
    print("       ↓")
    print("  Legendre + Bessel + Signal Features")
    print("       ↓")
    print("  Qiskit Quantum Encoding")
    print("       ↓")
    print("  Artificial Quantum Hardware Errors")
    print("       ↓")
    print("  Noisy Measurements")
    print("       ↓")
    print("  PennyLane Trainable Quantum Layer")
    print("       ↓")
    print("  PyTorch ML Decoder")
    print("       ↓")
    print("  Physical Feature Reconstruction + Classification")

    dataset = build_dataset()

    # Stratified split so all physical classes remain represented.
    (
        x_train_full,
        x_test,
        y_train_full,
        y_test,
        labels_train_full,
        labels_test,
    ) = train_test_split(
        dataset.noisy_quantum_features,
        dataset.physics_features,
        dataset.labels,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=dataset.labels,
    )

    (
        x_train,
        x_val,
        y_train,
        y_val,
        labels_train,
        labels_val,
    ) = train_test_split(
        x_train_full,
        y_train_full,
        labels_train_full,
        test_size=0.20,
        random_state=SEED,
        stratify=labels_train_full,
    )

    # Standardize target physical features using training data only.
    scaler = StandardScaler()

    y_train_scaled = scaler.fit_transform(y_train)
    y_val_scaled = scaler.transform(y_val)
    y_test_scaled = scaler.transform(y_test)

    x_train_tensor = torch.tensor(
        x_train,
        dtype=torch.float32,
    )

    x_test_tensor = torch.tensor(
        x_test,
        dtype=torch.float32,
    )

    x_val_tensor = torch.tensor(
        x_val,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train_scaled,
        dtype=torch.float32,
    )

    y_val_tensor = torch.tensor(
        y_val_scaled,
        dtype=torch.float32,
    )

    y_test_tensor = torch.tensor(
        y_test_scaled,
        dtype=torch.float32,
    )

    labels_val_tensor = torch.tensor(
        labels_val,
        dtype=torch.long,
    )

    labels_train_tensor = torch.tensor(
        labels_train,
        dtype=torch.long,
    )

    labels_test_tensor = torch.tensor(
        labels_test,
        dtype=torch.long,
    )

    print("\n[2/4] Dataset summary")
    print(f"    Samples: {len(dataset.physics_features)}")
    print(f"    Qubits: {N_QUBITS}")
    print(f"    Quantum states: {2 ** N_QUBITS}")
    print(f"    Physics features: {FEATURE_DIM}")
    print(f"    Quantum error probability: {QISKIT_ERROR_PROB}")
    print(f"    Phase damping: {PHASE_DAMPING}")

    model = QuantumInformedPhysicsAI()

    train_model(
        model,
        x_train_tensor,
        y_train_tensor,
        labels_train_tensor,
        x_val_tensor,
        y_val_tensor,
        labels_val_tensor,
    )

    evaluate_model(
        model,
        x_test_tensor,
        y_test_tensor,
        labels_test_tensor,
        scaler,
    )

    # Demonstrate recovery on a fresh, unseen physical signal.
    unseen_signal = generate_random_physical_signal(
        np.random.choice(PATTERNS)
    )

    demonstrate_single_recovery(
        model,
        unseen_signal,
        scaler,
    )

    # Save trained model.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_dim": FEATURE_DIM,
            "n_qubits": N_QUBITS,
            "patterns": PATTERNS,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        "formulon_quantum_physics_ai_v1.pt",
    )

    print("\nSaved:")
    print("    formulon_quantum_physics_ai_v1.pt")

    print("\nFormulon v1.0 prototype complete.")


if __name__ == "__main__":
    main()
