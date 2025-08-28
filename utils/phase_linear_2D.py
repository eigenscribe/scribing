# utils/phase_linear_2D.py
# ------------------------------------------------------------
# Plot phase portrait for a 2D Linear System
# ------------------------------------------------------------
"""
Utilities for drawing phase-portraits of 2-D linear systems.

Typical usage in a Jupyter notebook:

    >>> from utils.phase import phase_portrait, set_plot_style
    >>> set_plot_style("dark_background")               # optional - pick a style that matches your site
    >>> phase_portrait(A_matrix)                        # creates its own figure
    >>> # or, for a grid of several sysetms:
    >>> fig, axes = plt.subplots(1, 3, figsize=(18,5))
    >>> for A, ax in zip([A1, A2, A3], axes):
    ...     phase_portrait(A, ax=ax)

The function automatically:
* computes eigenvalues/eigenvectors,
* classifies the fixed point,
* builds a descriptive title,
* shades the equilibrium point according to stability,
* raises a friendly error (with an emoji) if the matrix is singular.

"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np 
from numpy.linalg import eig
from numpy.typing import NDArray
from typing import Literal, Optional, Tuple

# ------------------------------------------------------------
# Type aliases (helps IDEs and static analysers)
# ------------------------------------------------------------
Matrix2x2 = NDArray[np.float64]         # 2x2 array of floats
StyleName = str                         # any Matplotlib style string



# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def set_plot_style(style_name: StyleName) -> None:
    """
    Apply a Matplotlib style globally.

    Parameters
    ----------
    style_name
        Name of a built-in styole (e.g., "default", "ggplot", "dark_background")
        or the path to a custom * .mplstyle* file.
    """
    plt.style.use(style_name)


# ------------------------------------------------------------
# Core routine
# ------------------------------------------------------------
def phase_portrait(
        A: Matrix2x2,
        *,
        ax: Optional[plt.Axes] = None,
        grid_density: int = 20,
        domain: Tuple[float, float] = (-5.0, 5.0),
) -> None:
    """
    Plot the phase portrait of xdot = A*x for a 2-by-2  matrix ``A``.

    Features
    * computes eigenvalues/eigenvectors,
    * handles both non-singular (isolated equilibrium at origin) and singular cases (line or plane of fixed points)
    * classifies the fixed point (stable node, unstable node, saddle, centre, spiral),
    * builds a title that mentions the classification and the eigenvalues,
    * shades the equilibrium point:
        - solid fill    -> stable node,
        - half-filled   -> saddle,
        - outline only  -> unstable node, 
        - dotted edge   -> centre,

    """

    # --------------------------------------------------------------
    # 1️⃣ Validate input
    # --------------------------------------------------------------
    if A.shape !=(2, 2):
        raise ValueError("🚫 A must be a 2x2 matrix.")
    
    detA = np.linalg.det(A)
    singular = np.isclose(detA, 0.0)

    # --------------------------------------------------------------
    # 2️⃣ Eigendecomposition
    # --------------------------------------------------------------
    evals, evecs = eig(A)                     # evals: (2,), evecs: (2,2)

    # --------------------------------------------------------------
    # 3️⃣ Prepare the Axes (create if necessary)
    # --------------------------------------------------------------
    created_here = False
    if ax is None:
        fix, ax = plt.subplots(figsize=(6, 6))
        created_here = True

    # --------------------------------------------------------------
    # 4️⃣ Vector fields (streamlines)
    # --------------------------------------------------------------
    xmin, xmax = domain
    x, y = np.meshgrid(
        np.linspace(xmin, xmax, grid_density),
        np.linspace(xmin, xmax, grid_density),
    )
    u = A[0, 0] * x + A[0, 1] * y   # dx/dt
    v = A[1, 0] * x + A[1, 1] * y   # dy/dt
    ax.streamplot(x, y, u, v, color="b", linewidth=1)

    # --------------------------------------------------------------
    # 5️⃣ Handle singular vs. nonsigular cases
    # --------------------------------------------------------------
    if singular:
        # Singular case: fixed points are not isolated
        rank = np.linalg.matrix_rank(A)

        if rank == 0:
            node_type = "everywhere fixed (A=0)"
            ax.text(0, 0, "All points fixed", ha="center", 
                    va="center", color="gold", fontsize=12)
            
        elif rank == 1:
            node_type = "line of fixed points"
            # nullspace vector = eigenvector for eigenvalue ~ 0
            nullvec = np.real(evecs[:, np.isclose(evals, 0.0)])
            if nullvec.shape[1] > 0:
                nv = nullvec[:, 0]
                t = np.linspace(xmin, xmax, 200)
                ax.plot(t * nv[0], t * nv[1], "gold", lw=2, label="fixed line")

        # Draw eigenvectors (optional for singular too)
        for vec in evecs.T:
            ax.quiver(
                0, 0,
                3 * np.real(vec[0]),
                3 * np.real(vec[1]),
                scale=1, scale_units="xy", angles="xy",
                color="r", width=0.005,
            )

        eig_str = ", ".join(f"{ev:.3g}" for ev in evals)
        ax.set_title(f"Singular: {node_type}\nEigenvalues: [{eig_str}]")

    else:
        # ------------------------------------------------------------
        # 6️⃣ Non-singualr case: Classify fixed point
        # ------------------------------------------------------------
        evals_real_part = np.real(evals)

        def _classify(r: NDArray[np.float64]) -> str:
            pos = np.sum(r > 0)
            neg = np.sum(r < 0)

            if pos == 0 and neg == 0:
                return "center"
            if pos == 0 and neg == 2:
                return "stable node"
            if pos == 2 and neg == 0:
                return "unstable node"
            if pos == 1 and neg == 1:
                return "saddle"
            if np.any(np.iscomplex(evals)):
                return "stable spiral" if np.mean(r) < 0 else "unstable spiral"
            return "saddle"
        
        node_type = _classify(evals_real_part)

        # Draw eigenvectors
        for vec in evecs.T:
            ax.quiver(
                0, 0, 
                3 * np.real(vec[0]),
                3 * np.real(vec[1]),
                scale=1, scale_units="xy", angles="xy",
                color="r", width=0.005,
            )

        # Equilibrium marker
        from matplotlib.patches import Circle, Wedge
        radius = 0.25

        if node_type == "stable node":
            marker = Circle((0, 0), radius, facecolor="gold", edgecolor="k", lw=1.5)
        elif node_type == "unstable node":
            marker = Circle((0, 0), radius, facecolor="none", edgecolor="k", lw=1.5)
        elif node_type == "saddle":
            marker = Wedge((0, 0), radius, 0, 180,
                           facecolor="gold", edgecolor="k", lw=1.5)
            ax.add_patch(marker)
            marker = Wedge((0, 0), radius, 180, 360,
                           facecolor="none", edgecolor="k", lw=1.5)
            ax.add_patch(marker)
            marker = Wedge((0, 0), radius, 180, 360,
                           facecolor="none", edgecolor="k", lw=1.5)
        elif node_type.endswith("spiral"):
            marker = Circle((0, 0), radius, facecolor="gold", 
                            edgecolor="k", hatch="xx", lw=1.5)
        else:   # center
            marker = Circle((0, 0), radius, facecolor="none",
                            edgecolor="k", linestyle=":", lw=1.5)
            
        ax.add_patch(marker)

        eig_str = ", ".join(f"{ev:.3g}" for ev in evals)
        ax.set_title(f"{node_type.title()} - eigenvalues: [{eig_str}]")

        # ------------------------------------------------------------
        # 🔟 Cosmetics
        # ------------------------------------------------------------
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(xmin, xmax)
        if created_here:
            plt.tight_layout()
            plt.show()

# ------------------------------------------------------------
# Habit 2 - def main() -> None:
# ------------------------------------------------------------
def main() -> None:
    """
    Simple demo when running this module directly.
    Plots phase portraits for a few example systems:
    - saddle
    - stable spiral
    -singular case (line of ixed points)
    """

    # 🔵 Example matrices
    A1 = np.array([[2, 1],
                   [1, -1]], dtype=float)       # Saddle
    A2 = np.array([[0, -1],
                   [1, -1]], dtype=float)       # Stable spiral
    A3 = np.array([[1, 0],
                   [0, 0]], dtype=float)        # Singular, line of fixed points (rank(A3)=1)
    A4 = np.array([[0, 0],
                   [0, 0]], dtype=float)        # Singular, fixed points forall Re^2
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for A, ax in zip([A1, A2, A3, A4], axes):
        phase_portrait(A, ax=ax)

    plt.suptitle("Demo: 2D Linear System Phase Portraits", fontsize=14)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Habit 1 - if __name__ == "__main__":
# ------------------------------------------------------------
if __name__ == '__main__':
    main()
