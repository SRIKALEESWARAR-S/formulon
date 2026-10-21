# Contributing to Formulon Physics

Thank you for your interest in contributing to Formulon Physics!

We welcome contributions that improve the scientific correctness, reliability, documentation, accessibility, and maintainability of the project.

## Ways to Contribute

You can contribute by:

- Fixing bugs and improving existing implementations.

- Adding or improving automated tests.

- Improving documentation and usage examples.

- Proposing new physics formulas with scientific references.

- Improving validation of inputs, units, and assumptions.

- Improving packaging and development workflows.

## Development Setup

### Requirements

- Python 3.10 or newer.

- Git.

- A Python virtual environment.

- A supported platform and compatible scientific Python dependencies.

The project primarily targets Linux. Windows contributors may encounter platform-specific issues.

### 1. Fork and Clone

Fork the repository on GitHub, then clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/formulon.git

cd formulon
```
### 2. Create a Virtual Environment

On Linux or macOS:
```bash
python -m venv .venv

source .venv/bin/activate
```
On Windows PowerShell:
```powershell
py -3.11 -m venv .venv

.\.venv\\Scripts\\Activate.ps1
```
Python 3.11 is an example supported version. Use another compatible Python version if necessary.

### 3. Install Dependencies

Install the project in editable mode:
```bash
python -m pip install -e .
```
Install pytest:
```bash
python -m pip install pytest
```
Optional quantum dependencies:
```bash
python -m pip install -e ".[quantum]"
```
Quantum dependencies are not required for most core contributions.

## Running Tests

Run the test suite from the repository root:
```bash
python -m pytest tests
```
To run an individual test module:
```bash
python -m pytest tests/test_formula_catalog.py
```
Some tests or imports may require optional dependencies. Report any installation or test failures with your Python version, operating system, and relevant error output.

## Working on an Issue

Before starting significant changes:

1. Search existing issues and pull requests.

2. Open an issue if the problem has not been reported.

3. Discuss large features or architectural changes with maintainers.

4. Create a focused branch for your contribution.

Example:
```bash
git switch -c fix/short-description
```
Keep unrelated changes out of your pull request.

## Scientific Formula Contributions

When adding or modifying a physics formula:

1. Explain the physical principle and intended use.

2. Provide a reliable scientific or textbook reference.

3. Document the equation and define all variables.

4. Specify units, preferably SI units for numeric APIs.

5. Describe assumptions and valid input domains.

6. Include tests for representative and boundary cases.

7. Verify expected results against trusted references or independently derived calculations.

8. Update relevant documentation and formula metadata where applicable.

Do not introduce arbitrary physical limits without mathematical or model-specific justification.

Scientific correctness is important. Maintainers may request additional evidence or review before accepting formula changes.

## Code Quality

- Follow existing code conventions.

- Use descriptive function and variable names.

- Add docstrings for public functions.

- Keep changes small and understandable.

- Avoid unnecessary dependencies.

- Include tests for bug fixes and new functionality.

- Avoid introducing breaking API changes without prior discussion.

## Pull Request Guidelines

Before opening a pull request:

1. Review your changes using `git diff`.

2. Run relevant tests where possible.

3. Check that only intended files are staged.

4. Write a clear pull request title and description.

5. Link related issues where applicable.

6. Explain testing performed and any known limitations.

Maintainers review contributions based on project scope, technical quality, scientific correctness, and available review capacity.

## Reporting Bugs

Use the [GitHub issue tracker](https://github.com/SRIKALEESWARAR-S/formulon/issues) for ordinary bugs and feature requests.

Include:

- A description of the problem.

- Steps to reproduce.

- Expected and actual behavior.

- Python and package versions.

- Operating system.

- Relevant error messages.

Do not publicly disclose sensitive security vulnerabilities. Contact the maintainers to request an appropriate private reporting channel until a security policy is published.

## Community Conduct

Contributors are expected to communicate respectfully and provide constructive feedback.

A dedicated Code of Conduct is being prepared and should be consulted once adopted by the project maintainers.

Thank you for helping improve Formulon Physics!
