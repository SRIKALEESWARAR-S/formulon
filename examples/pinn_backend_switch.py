"""Same PINN problem on every installed autodiff backend (NumPy always works)."""
import formulon
from formulon.pinn import get_problem, fit, evaluate, available_autodiff

problem = get_problem("heat_1d")
print("installed autodiff backends:", available_autodiff())

model, info = fit(problem, backend="numpy")               # NumPy-only PIELM solver
print("numpy  rel-L2:", f"{evaluate(problem, model, backend='numpy')['rel_l2']:.2e}")

if "torch" in available_autodiff():                        # gradient-based PINN
    model, hist = fit(problem, backend="torch", epochs=3000)
    print("torch  rel-L2:", f"{evaluate(problem, model, backend='torch')['rel_l2']:.2e}")

print(formulon.backend_status())
