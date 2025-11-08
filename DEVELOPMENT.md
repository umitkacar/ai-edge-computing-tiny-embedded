# Development Guide

This guide covers the development setup and workflows for the AI Edge TinyML project.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/umitkacar/ai-edge-computing-tiny-embedded.git
cd ai-edge-computing-tiny-embedded

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
hatch run test

# Run tests with coverage
hatch run test-cov

# Run specific test file
pytest tests/test_quantization.py -v

# Run with markers
pytest -m unit  # Only unit tests
pytest -m "not slow"  # Skip slow tests
```

### Code Quality

```bash
# Run linter
hatch run lint

# Auto-fix linting issues
ruff check --fix src tests

# Format code
hatch run format

# Check formatting without changes
hatch run format-check

# Type checking
hatch run type-check

# Run all checks
hatch run all
```

### Pre-commit Hooks

Pre-commit hooks run automatically before each commit. To run manually:

```bash
# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files
pre-commit run mypy --all-files

# Update hooks to latest version
pre-commit autoupdate
```

## 🛠️ Development Tools

### Hatch Scripts

The project uses Hatch for managing development environments. Available scripts:

- `hatch run test` - Run pytest
- `hatch run test-cov` - Run pytest with coverage report
- `hatch run lint` - Run ruff linter
- `hatch run format` - Format code with black
- `hatch run format-check` - Check code formatting
- `hatch run type-check` - Run mypy type checker
- `hatch run all` - Run all quality checks

### Coverage Reports

After running tests with coverage:

```bash
# View HTML coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows

# View coverage summary
coverage report

# Generate XML for CI
coverage xml
```

## 🔧 Project Structure

```
ai-edge-computing-tiny-embedded/
├── src/ai_edge_tinyml/          # Main package
│   ├── __init__.py              # Package exports
│   ├── model_optimizer.py       # Model optimization utilities
│   ├── quantization.py          # Quantization implementations
│   └── utils.py                 # Helper functions
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── conftest.py              # Shared pytest fixtures
│   ├── test_quantization.py    # Quantization tests
│   └── test_utils.py            # Utility function tests
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI/CD
├── pyproject.toml               # Project configuration
├── .pre-commit-config.yaml      # Pre-commit hooks
├── .gitignore                   # Git ignore patterns
└── README.md                    # Project overview
```

## 📊 Code Quality Standards

### Type Annotations

All code must be fully type-annotated:

```python
from typing import Protocol
import numpy as np
import numpy.typing as npt

def process_weights(
    weights: npt.NDArray[np.float32],
    threshold: float = 0.01,
) -> npt.NDArray[np.float32]:
    """Process model weights with given threshold."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_accuracy(
    prediction: npt.NDArray[np.int_],
    target: npt.NDArray[np.int_],
) -> float:
    """Calculate classification accuracy.

    Args:
        prediction: Predicted class labels
        target: Target class labels

    Returns:
        Accuracy as a percentage (0-100)

    Raises:
        ValueError: If arrays have different shapes or are empty
    """
    ...
```

### Error Handling

Always validate inputs and provide clear error messages:

```python
if weights.size == 0:
    msg = "Cannot quantize empty weights array"
    raise ValueError(msg)
```

### Testing

Write comprehensive tests with parametrization:

```python
@pytest.mark.parametrize(
    ("mode", "expected_dtype"),
    [
        (QuantizationMode.INT8, np.int8),
        (QuantizationMode.INT4, np.int8),
        (QuantizationMode.FLOAT16, np.float16),
    ],
)
def test_quantize_modes(
    float32_array: npt.NDArray[np.float32],
    mode: QuantizationMode,
    expected_dtype: type,
) -> None:
    """Test different quantization modes."""
    ...
```

## 🔍 Troubleshooting

### Mypy Errors

If mypy reports errors:

```bash
# Run mypy with verbose output
mypy --show-error-codes src/

# Ignore specific errors (as last resort)
# type: ignore[error-code]
```

### Pre-commit Hook Failures

If pre-commit hooks fail:

```bash
# Skip hooks (not recommended)
git commit --no-verify

# Fix issues manually
pre-commit run --all-files
```

### Import Errors

Make sure the package is installed in development mode:

```bash
pip install -e ".[dev]"
```

## 📝 Commit Guidelines

Follow conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Build/tooling changes

Example:

```
feat(quantization): add dynamic quantization support

- Implement dynamic INT8 quantization
- Add tests for dynamic mode
- Update documentation

Closes #123
```

## 🚀 Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag: `git tag v0.1.0`
4. Push tag: `git push origin v0.1.0`
5. Build package: `python -m build`
6. Upload to PyPI: `twine upload dist/*`

## 📚 Additional Resources

- [Hatch Documentation](https://hatch.pypa.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run all quality checks: `hatch run all`
5. Commit with conventional commits
6. Push and create a pull request

## 📄 License

MIT License - see LICENSE file for details
