import numpy as np
from sympy import symbols, diff, solve, N, exp

# Define the symbolic variables
x_sym, y_sym = symbols('x y')

x0 = 0
y0 = 0

def f(x, y):
    return 4*x - x**2 + 6*y - y**2

# Get the variables from the objective function
def get_variables(func):
    variables = list(func.free_symbols)
    variables.sort(key=lambda var: str(var))  # Sort variables by their string representation
    return variables

# Gradient of the objective function
def gradient_dynamic_f(func, var_values):
    variables = get_variables(func)  # Get the list of variables
    gradient_vals = []
    gradient_formula = []
    for var in variables:
        partial_derivative = diff(func, var)  # Calculate the partial derivative
        gradient_formula.append(partial_derivative)
        partial_derivative_value = partial_derivative.evalf(subs=var_values)  # Evaluate the partial derivative at the given values
        gradient_vals.append(N(partial_derivative_value, chop=True))  # Use N() to handle small values
    print(gradient_formula)
    return np.array([gradient_vals, gradient_formula])



def find_optimal_t(x, y, grad):
    if np.all(grad == 0):
        # If the gradient is zero, we are at a critical point
        return 0, x, y

    t = symbols('t')
    x_t = x + t * grad[0]
    y_t = y + t * grad[1]
    f_t = f(x_t, y_t)
    df_dt = diff(f_t, t)
    t_opt = solve(df_dt, t)[0]  # Extract the optimal t value
    x_opt = x + t_opt * grad[0]  # Calculate the latest value of x using the optimal t
    y_opt = y + t_opt * grad[1]  # Calculate the latest value of y using the optimal t
    # print(x_opt,y_opt)
    return t_opt, x_opt, y_opt

def optimize(tol=0.0001, max_iter=1000):
    x, y = 0, 0
    iteration = 0

    while iteration < max_iter:
        var_values = {"x": x, "y": y}
        new_grad = gradient_dynamic_f(f(x_sym,y_sym), var_values = var_values)[0]
        # print(new_grad)
        t_opt, x_new, y_new = find_optimal_t(x, y, new_grad)  # Get the new values directly

        variables = get_variables(f(x_sym, y_sym))

        converged = True  # Initialize converged flag
        
        for var in variables:
            old_val = var_values[str(var)]
            new_val = locals()[str(var) + "_new"]  # Get the new value of the variable dynamically
            
            # Calculate the gap for the current variable
            gap = abs(new_val - old_val)
            
            # Check if the gap exceeds the tolerance for this variable
            if gap >= tol:
                converged = False
                break
        
        
        if converged:
            print("Converged in", {iteration + 1}, "iterations.")
            break
        # print(x,y)
        x, y = x_new, y_new
        iteration += 1
    else:
        print("Maximum number of iterations reached.")

    return x, y, t_opt

# Run the optimization
x_opt, y_opt, t_opt = optimize()

print(x_opt,y_opt)