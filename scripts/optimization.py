import numpy as np
from sympy import symbols, diff, solve, N
import sys

# Define the symbolic variables
x_sym, y_sym, z_sym, w_sym, v_sym = symbols('x y z w v')

if len(sys.argv) > 1:
    user_initial_values_array = [float(arg) for arg in sys.argv[1:]]
else:
    user_initial_values_array = []


def array_to_dict(variables, values):
    return {str(var): value for var, value in zip(variables, values)}


def initialize_var_values(variables, initial_values=None):
    if initial_values is None:
        initial_values = {}
    return {str(var): initial_values.get(str(var), 0) for var in variables}


def f(*args):
    x, y, z, w, v = args
    return 4*x - x**2 + 3*y - y**2 + 5*z - z**2 + 6*w - w**2 + 2*v - v**2

# Get the variables from the objective function
def get_variables(func):
    variables = list(func.free_symbols)
    variables.sort(key=lambda var: str(var))  # Sort variables by their string representation
    return variables

dict = array_to_dict(get_variables(f(x_sym,y_sym, z_sym, w_sym, v_sym)), user_initial_values_array)


# print(initialize_var_values(get_variables(f(x_sym,y_sym, z_sym, w_sym, v_sym)), dict))

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
    return np.array([gradient_vals, gradient_formula])

# print(gradient_dynamic_f(f(x_sym,y_sym,z_sym), var_values=var_values)[0], gradient_dynamic_f(f(x_sym,y_sym,z_sym), var_values=var_values)[1])

def find_optimal(var_values, grad, func):
    if np.all(grad == 0):
        # If the gradient is zero, we are at a critical point
        return 0, var_values

    t = symbols('t')
    variables = get_variables(func)

    # print(variables)

    # Generate the new variable values dynamically
    new_var_values = {str(var): var_values[str(var)] + t * grad[i] for i, var in enumerate(variables)}

    # print(new_var_values)

    f_t = f(*[new_var_values[str(var)] for var in variables])

    # print(f_t)

    df_dt = diff(f_t, t)

    # print(df_dt)

    t_opt_candidates = solve(df_dt, t)  # Solve for all t candidates

    # print(t_opt_candidates)

    # # Find the positive t value if exists, otherwise take the maximum t
    t_opt = max([t_val for t_val in t_opt_candidates if t_val > 0], default=t_opt_candidates[0])

    new_var_values_opt = {str(var): var_values[str(var)] + t_opt * grad[i] for i, var in enumerate(variables)}

    # print(new_var_values_opt)

    return t_opt, new_var_values_opt

# find_optimal(var_values, gradient_dynamic_f(f(x_sym,y_sym), var_values=var_values)[0], f(x_sym,y_sym))

def optimize(tol=0.0001, max_iter=1000):
    iteration = 0
    variables = get_variables(f(x_sym, y_sym, z_sym, w_sym, v_sym))
    var_values = initialize_var_values(variables)

    while iteration < max_iter:

        new_grad = gradient_dynamic_f(f(x_sym, y_sym, z_sym, w_sym, v_sym), var_values=var_values)[0]
        # print(new_grad)

        t_opt, new_var_values = find_optimal(var_values, new_grad, f(x_sym, y_sym, z_sym, w_sym, v_sym))
        # print(t_opt, new_var_values)


        # print(variables)

        converged = True  # Initialize converged flag

        # print(var_values)

        for var in variables:
            old_val = var_values[str(var)]
            new_val = new_var_values[str(var)]  # Get the new value of the variable dynamically
            # print(old_val, new_val)

            # Calculate the gap for the current variable
            gap = abs(new_val - old_val)

            # print(gap)

            #  Check if any of the gaps exceeds the tolerance
            if gap >= tol:
                converged = False
                break

        if converged:
            print("Converged in", {iteration + 1}, "iterations.")
            break
        # print(converged)

        var_values = new_var_values  # Update var_values with new values

        # print(new_var_values , var_values)

        iteration += 1
        # print(converged)
    else:
        print("Maximum number of iterations reached.")

    return var_values, t_opt


# Run the optimization
values_new, t_opt = optimize()
# optimize()

print("Optimal values:", values_new)
# print("Optimal t:", t_opt)