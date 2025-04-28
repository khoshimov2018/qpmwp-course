############################################################################
### QPMwP - Turnover Constraint Linearization
############################################################################

# --------------------------------------------------------------------------
# Student solution for Assignment 2
# --------------------------------------------------------------------------

# Standard library imports
from typing import Optional, Union

# Third party imports
import numpy as np
import pandas as pd

# Local modules imports
from optimization.quadratic_program import QuadraticProgram

def linearize_turnover_constraint(self, x_init: np.ndarray, to_budget=float('inf')) -> None:
    '''
    Linearize the turnover constraint in the quadratic programming problem.

    This method modifies the quadratic programming problem to include a linearized turnover constraint.

    Parameters:
    -----------
    x_init : np.ndarray
        The initial portfolio weights.
    to_budget : float, optional
        The maximum allowable turnover. Defaults to float('inf').

    Notes:
    ------
    - The method updates the problem's objective function coefficients, inequality constraints,
    equality constraints, and bounds to account for the turnover constraint.
    - The original problem data is overridden with the updated matrices and vectors.

    Examples:
    ---------
    >>> qp = QuadraticProgram(P, q, G, h, A, b, lb, ub, solver='cvxopt')
    >>> qp.linearize_turnover_constraint(x_init=np.array([0.1, 0.2, 0.3]), to_budget=0.05)
    '''
    # Dimensions
    n = len(self.problem_data.get('q'))
    m = 0 if self.problem_data.get('G') is None else self.problem_data.get('G').shape[0]
    p = 0 if self.problem_data.get('A') is None else self.problem_data.get('A').shape[0]
    
    # In the two-fold method, we introduce auxiliary variables z
    # Where z_i = |x_i - x_init_i|
    # The turnover constraint becomes sum(z_i) <= to_budget
    # This doubles the dimensionality (from n to 2n)
    
    # Extract current problem data
    P_orig = self.problem_data.get('P')
    q_orig = self.problem_data.get('q')
    G_orig = self.problem_data.get('G')
    h_orig = self.problem_data.get('h')
    A_orig = self.problem_data.get('A')
    lb_orig = self.problem_data.get('lb')
    ub_orig = self.problem_data.get('ub')
    
    # Update the coefficients of the objective function
    # The auxiliary variables z don't contribute to the objective
    P = np.zeros((2*n, 2*n))
    P[:n, :n] = P_orig  # Upper-left block contains original P
    
    q = np.zeros(2*n)
    q[:n] = q_orig  # First n elements contain original q
    
    # Update the equality constraints
    # For each auxiliary variable z_i, we need constraints to enforce z_i >= x_i - x_init_i and z_i >= x_init_i - x_i
    if A_orig is not None:
        A = np.zeros((p, 2*n))
        A[:, :n] = A_orig  # Apply original equality constraints to x
    else:
        A = None
    
    # Update the inequality constraints
    # We need to add constraints for:
    # 1. z_i >= x_i - x_init_i
    # 2. z_i >= x_init_i - x_i
    # 3. sum(z_i) <= to_budget (turnover constraint)
    
    # Start with original constraints
    if G_orig is not None:
        # Create new G matrix with original constraints applied to x
        new_G = np.zeros((m + 2*n + 1, 2*n))
        new_G[:m, :n] = G_orig
        new_h = h_orig.copy() if h_orig is not None else np.array([])
    else:
        new_G = np.zeros((2*n + 1, 2*n))
        new_h = np.array([])
    
    # Add constraints: z_i >= x_i - x_init_i
    for i in range(n):
        row = m + i
        new_G[row, i] = 1       # Coefficient for x_i
        new_G[row, n + i] = -1  # Coefficient for z_i
        new_h = np.append(new_h, x_init[i])
    
    # Add constraints: z_i >= x_init_i - x_i
    for i in range(n):
        row = m + n + i
        new_G[row, i] = -1      # Coefficient for x_i
        new_G[row, n + i] = -1  # Coefficient for z_i
        new_h = np.append(new_h, -x_init[i])
    
    # Add turnover constraint: sum(z_i) <= to_budget
    turnover_row = new_G.shape[0] - 1
    new_G[turnover_row, n:2*n] = 1  # Sum of all z_i
    new_h = np.append(new_h, to_budget)
    
    G = new_G
    h = new_h
    
    # Update lower and upper bounds
    if lb_orig is not None:
        lb = np.zeros(2*n)
        lb[:n] = lb_orig  # Original lower bounds for x
        # z_i >= 0 since they represent absolute values
        lb[n:2*n] = 0
    else:
        lb = None
        
    if ub_orig is not None:
        ub = np.zeros(2*n)
        ub[:n] = ub_orig  # Original upper bounds for x
        # z_i can be as large as necessary (limited by other constraints)
        ub[n:2*n] = float('inf')
    else:
        ub = None
    
    # Override the original matrices (notice: b does not change)
    self.update_problem_data({
        'P': P,
        'q': q,
        'G': G,
        'h': h,
        'A': A,
        'lb': lb,
        'ub': ub
    })

    return None

# Add the method to QuadraticProgram class
QuadraticProgram.linearize_turnover_constraint = linearize_turnover_constraint