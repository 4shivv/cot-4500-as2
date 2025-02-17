import numpy as np

def neville_interpolation(x_points, y_points, x):
    n = len(x_points)
    Q = np.zeros((n, n))
    Q[:, 0] = y_points

    for i in range(1, n):
        for j in range(n - i):
            Q[j, i] = ((x - x_points[j + i]) * Q[j, i - 1] - (x - x_points[j]) * Q[j + 1, i - 1]) / (x_points[j] - x_points[j + i])
    
    return Q[0, n - 1]

def newton_forward_interpolation(x_points, y_points):
    n = len(x_points)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y_points

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i, j] = (diff_table[i + 1, j - 1] - diff_table[i, j - 1]) / (x_points[i + j] - x_points[i])

    return diff_table

def newton_forward_eval(diff_table, x_points, x):
    n = len(x_points)
    approx = diff_table[0, 0]
    term = 1.0

    for i in range(1, n):
        term *= (x - x_points[i - 1])
        approx += term * diff_table[0, i]

    return approx

def hermite_interpolation(x_points, y_points, y_derivatives):
    n = len(x_points) * 2
    H = np.zeros((n, n))
    Q = np.zeros((n, n))
    z = np.zeros(n)
    f = np.zeros(n)

    for i in range(len(x_points)):
        z[2 * i] = z[2 * i + 1] = x_points[i]
        f[2 * i] = f[2 * i + 1] = y_points[i]
        Q[2 * i, 0] = Q[2 * i + 1, 0] = y_points[i]
        Q[2 * i + 1, 1] = y_derivatives[i]
        if i != 0:
            Q[2 * i, 1] = (Q[2 * i, 0] - Q[2 * i - 1, 0]) / (z[2 * i] - z[2 * i - 1])

    for j in range(2, n):
        for i in range(n - j):
            Q[i, j] = (Q[i + 1, j - 1] - Q[i, j - 1]) / (z[i + j] - z[i])

    return Q

def cubic_spline_interpolation(x_points, y_points):
    n = len(x_points)
    h = np.diff(x_points)
    b = (np.diff(y_points) / h)
    
    A = np.zeros((n, n))
    rhs = np.zeros(n)
    
    A[0, 0] = 1
    A[-1, -1] = 1

    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 3 * (b[i] - b[i - 1])

    coeffs = np.linalg.solve(A, rhs)
    return A, rhs, coeffs

if __name__ == "__main__":
    # Question 1
    x_points = np.array([3.6, 3.8, 3.9])
    y_points = np.array([1.675, 1.436, 1.318])
    print(neville_interpolation(x_points, y_points, 3.7))

    # Question 2 & 3
    x_points = np.array([7.2, 7.4, 7.5, 7.6])
    y_points = np.array([23.5492, 25.3913, 26.8224, 27.4589])
    diff_table = newton_forward_interpolation(x_points, y_points)
    print(diff_table)

    print(newton_forward_eval(diff_table, x_points, 7.3))

    # Question 4
    x_points = np.array([3.6, 3.8, 3.9])
    y_points = np.array([1.675, 1.436, 1.318])
    y_derivatives = np.array([-1.195, -1.188, -1.182])
    print(hermite_interpolation(x_points, y_points, y_derivatives))

    # Question 5
    x_points = np.array([2, 5, 8, 10])
    y_points = np.array([3, 5, 7, 9])
    A, b, x = cubic_spline_interpolation(x_points, y_points)
    print(A)
    print(b)
    print(x)
