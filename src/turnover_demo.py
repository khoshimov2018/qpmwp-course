############################################################################
### QPMwP - Turnover Constraint Demo
############################################################################

# --------------------------------------------------------------------------
# Student solution for Assignment 2
# --------------------------------------------------------------------------

# Standard library imports
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Third party imports
import numpy as np
import pandas as pd

# Local modules imports
from src.estimation.covariance import Covariance
from src.estimation.expected_return import ExpectedReturn
from src.optimization.constraints import Constraints
from src.optimization.quadratic_program import QuadraticProgram
from src.helper_functions import load_data_msci
from src.optimization.turnover import linearize_turnover_constraint  # This will monkey-patch the QuadraticProgram class

def main():
    """
    Demo of turnover constraint linearization
    """
    print("TURNOVER CONSTRAINT DEMO")
    print("========================")
    
    # Load data
    print("\n1. Loading data...")
    N = 10
    data_path = os.path.join(project_root, 'data')
    data = load_data_msci(path=data_path, n=N)
    X = data['return_series']
    
    # Compute expected return and covariance matrix
    print("\n2. Computing expected return and covariance...")
    q = ExpectedReturn(method='geometric').estimate(X=X, inplace=False)
    P = Covariance(method='pearson').estimate(X=X, inplace=False)
    
    # Set up constraints
    print("\n3. Setting up constraints...")
    constraints = Constraints(ids=X.columns.tolist())
    constraints.add_budget(rhs=1, sense='=')
    constraints.add_box(lower=0.0, upper=1.0)
    GhAb = constraints.to_GhAb()
    
    # Create quadratic program
    print("\n4. Creating quadratic program...")
    qp = QuadraticProgram(
        P=P.to_numpy(),
        q=q.to_numpy() * 0,  # Minimize variance only
        G=GhAb['G'],
        h=GhAb['h'],
        A=GhAb['A'],
        b=GhAb['b'],
        lb=constraints.box['lower'].to_numpy(),
        ub=constraints.box['upper'].to_numpy(),
        solver='cvxopt',
    )
    
    # Prepare initial weights (equal weighting)
    print("\n5. Setting up initial weights...")
    x_init = pd.Series([1/X.shape[1]]*X.shape[1], index=X.columns)
    
    # Add turnover constraint and solve
    print("\n6. Adding turnover constraint and solving...")
    qp.linearize_turnover_constraint(x_init=x_init, to_budget=0.5)
    qp.solve()
    
    # Check turnover
    print("\n7. Checking results...")
    solution = qp.results.get('solution')
    ids = constraints.ids
    weights = pd.Series(solution.x[:len(ids)], index=ids)
    
    turnover = np.abs(weights - x_init).sum()
    print(f"Turnover: {turnover}")
    print(f"Turnover limit: 0.5")
    
    # Check if constraint is respected
    if turnover <= 0.5:
        print("SUCCESS: Turnover constraint is respected!")
    else:
        print("ERROR: Turnover constraint is violated!")
    
    print("\nWeights:")
    print(weights)
    
    print("\nDONE!")

if __name__ == "__main__":
    main()