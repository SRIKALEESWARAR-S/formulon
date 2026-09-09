"""Release QA checks for Formulon Physics."""
from pathlib import Path
import ast, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
errors=[]
for path in (ROOT/'src').rglob('*.py'):
    try: ast.parse(path.read_text())
    except SyntaxError as e: errors.append(f'{path}: {e}')
if errors:
    print('SYNTAX FAIL'); print('\n'.join(errors)); sys.exit(1)
import formulon
from formulon.formula_catalog import FORMULA_COUNT
print(f'Formulon {formulon.__version__}: {FORMULA_COUNT} public formulas')
print('QA PASS')
