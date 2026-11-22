"""Quantum adders on whichever backend you prefer."""
from formulon.quantum_computing import add, available_backends, use_backend, half_adder, set_backend

print("backends:", available_backends())
for backend in available_backends():
    with use_backend(backend):
        print(f"{backend:10s} 5 + 9 =", add(5, 9, n_bits=4), "| QFT adder (mod 16):", add(5, 9, n_bits=4, method="qft"))
print(half_adder().draw())
