import numpy as np

# Problem 1: Neville's Method
def neville_method(x_values, y_values, x):
    n = len(x_values)
    Q = np.zeros((n, n))
    Q[:, 0] = y_values

    for i in range(1, n):
        for j in range(n - i):
            Q[j, i] = ((x - x_values[j + i]) * Q[j, i - 1] + (x_values[j] - x) * Q[j + 1, i - 1]) / (x_values[j] - x_values[j + i])

    return Q[0, -1]

# Problem 2: Newton’s Forward Difference Table
def newton_forward_table(x_values, y_values):
    n = len(x_values)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y_values

    for i in range(1, n):
        for j in range(n - i):
            diff_table[j, i] = diff_table[j + 1, i - 1] - diff_table[j, i - 1]

    return diff_table

# Problem 3: Newton's Forward Interpolation
def newton_forward_interpolation(x_values, y_values, x):
    n = len(x_values)
    diff_table = newton_forward_table(x_values, y_values)
    h = x_values[1] - x_values[0]
    p = (x - x_values[0]) / h

    result = y_values[0]
    factorial = 1
    term = 1

    for i in range(1, n):
        factorial *= i
        term *= (p - (i - 1))
        result += (term * diff_table[0, i]) / factorial

    return result

# Problem 4: Hermite Interpolation
def hermite_interpolation(x_values, y_values, y_derivatives):
    n = len(x_values)
    size = 2 * n
    H = np.zeros((size, size))

    z = np.zeros(size)
    Q = np.zeros((size, size))

    for i in range(n):
        z[2 * i] = x_values[i]
        z[2 * i + 1] = x_values[i]
        Q[2 * i, 0] = y_values[i]
        Q[2 * i + 1, 0] = y_values[i]
        Q[2 * i + 1, 1] = y_derivatives[i]

        if i != 0:
            Q[2 * i, 1] = (Q[2 * i, 0] - Q[2 * i - 1, 0]) / (z[2 * i] - z[2 * i - 1])

    for i in range(2, size):
        for j in range(2, i + 1):
            Q[i, j] = (Q[i, j - 1] - Q[i - 1, j - 1]) / (z[i] - z[i - j])

    return Q

# Problem 5: Cubic Spline Interpolation
def cubic_spline_matrix(x_values, y_values):
    n = len(x_values)
    A = np.zeros((n, n))
    b = np.zeros(n)

    for i in range(1, n - 1):
        A[i, i - 1] = x_values[i] - x_values[i - 1]
        A[i, i] = 2 * (x_values[i + 1] - x_values[i - 1])
        A[i, i + 1] = x_values[i + 1] - x_values[i]
        b[i] = 3 * ((y_values[i + 1] - y_values[i]) / (x_values[i + 1] - x_values[i]) -
                    (y_values[i] - y_values[i - 1]) / (x_values[i] - x_values[i - 1]))

    A[0, 0] = A[-1, -1] = 1

    return A, b

if __name__ == "__main__":
    # Problem 1
    x_vals = [3.6, 3.8, 3.9]
    y_vals = [1.675, 1.436, 1.318]
    print(neville_method(x_vals, y_vals, 3.7))

    # Problem 2
    x_vals = [7.2, 7.4, 7.5, 7.6]
    y_vals = [23.5492, 25.3913, 26.8224, 27.4589]
    print(newton_forward_table(x_vals, y_vals))

    # Problem 3
    print(newton_forward_interpolation(x_vals, y_vals, 7.3))

    # Problem 4
    x_vals = [3.6, 3.8, 3.9]
    y_vals = [1.675, 1.436, 1.318]
    y_derivs = [-1.195, -1.188, -1.182]
    print(hermite_interpolation(x_vals, y_vals, y_derivs))

    # Problem 5
    x_vals = [2, 5, 8, 10]
    y_vals = [3, 5, 7, 9]
    A, b = cubic_spline_matrix(x_vals, y_vals)
    print(A)
    print(b)
