import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def solve(domain):
    # Define the domain size and grid spacing
    nx, ny = domain.shape
    dx, dy = (domain[0][0] - domain[nx][0]) / \
        nx, (domain[0][1] - domain[nx][1]) / ny

    # Define the right-hand side function f(x, y)
    def f(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    X, Y = np.meshgrid(x, y, indexing='ij')

    # Evaluate the right-hand side function on the grid
    F = f(X, Y)

    # Flatten the grid and the right-hand side function
    F_flat = F.flatten()

    # Define the Laplacian operator using finite differences
    def laplacian_2d(nx, ny, dx, dy):
        N = nx * ny
        diagonals = np.zeros((5, N))

        # Main diagonal
        diagonals[2, :] = -2 * (1/dx**2 + 1/dy**2)

        # Off-diagonals for x-direction
        diagonals[1, :-1] = 1 / dx**2
        diagonals[3, 1:] = 1 / dx**2

        # Off-diagonals for y-direction
        diagonals[0, :-nx] = 1 / dy**2
        diagonals[4, nx:] = 1 / dy**2

        # Create the sparse matrix
        A = diags(diagonals, [-nx, -1, 0, 1, nx], shape=(N, N))
        return A

    # Create the Laplacian matrix
    A = laplacian_2d(nx, ny, dx, dy)

    # Solve the linear system
    u_flat = spsolve(A, F_flat)

    print(u_flat)
