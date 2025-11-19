import numpy as np

def lu(A):
    """
    Performs LU Decomposition using Doolittle’s method with partial pivoting.
    Returns P, L, U such that P @ A = L @ U.
    """
    A = A.astype(float)  # Ensure floating point
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    P = np.eye(n)

    # Copy of A to avoid overwriting
    A_copy = A.copy()

    for k in range(n):
        pivot = np.argmax(np.abs(A_copy[k:, k])) + k
        if pivot != k:
            A_copy[[k, pivot], :] = A_copy[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]

        U[k, k:] = A_copy[k, k:]
        L[k, k] = 1.0

        for i in range(k + 1, n):
            L[i, k] = A_copy[i, k] / U[k, k]
            A_copy[i, k:] -= L[i, k] * U[k, k:]

    return P, L, U


if __name__ == "__main__":
    A = np.array([
        [2, 7, 1],
        [-2, 3, 0],
        [1, 5, 3]
    ], dtype=float)

    P, L, U = lu(A)

    print("P =\n", P)
    print("L =\n", L)
    print("U =\n", U)
    print("\nCheck PA ≈ LU:", np.allclose(P @ A, L @ U))
