# CLAUDE.md: Guidelines for the QPMwP Repository

## Build/Run Commands
- **Install Dependencies**: `pip install -r requirements.txt`
- **Run Examples**: `python examples/backtest_1.py`
- **Run Tests**: `pytest path/to/test_file.py`
- **Run Single Test**: `pytest path/to/test_file.py::test_function_name`

## Code Style Guidelines

### Imports
- Standard library imports first, then third-party imports, then local modules
- Group imports with blank lines between groups
- Within each group, use alphabetical order

### Naming Conventions
- Classes: PascalCase (`QuadraticProgram`, `BacktestService`)
- Functions/methods: snake_case (`load_data_msci`, `prepare_rebalancing`)
- Variables: snake_case (`rebalancing_date`, `return_series`)
- Constants: UPPER_CASE (`ALL_SOLVERS`, `SPARSE_SOLVERS`)

### Type Annotations
- Use Python type hints for function parameters and return values
- Use Optional[] for parameters that can be None
- Use Union[] for parameters that can be multiple types

### Docstrings
- Use Google-style docstrings with Parameters and Returns sections
- Document raised exceptions using the Raises section

### Error Handling
- Use explicit try/except blocks with specific exceptions
- Re-raise exceptions with context when appropriate

### Code Structure
- Classes should follow the property pattern for getters
- Use private attributes with leading underscore
- Include class-level docstrings for public classes