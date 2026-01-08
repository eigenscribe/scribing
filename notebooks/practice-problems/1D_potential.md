---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: qenv
    language: python
    name: python3
---

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import pandas as pd  # Import necessary libraries
import numpy as np
import sympy as sp
from IPython.display import display, Math

import sys

sys.path.append('/workspaces/scribing/')
from src.colors import Bcolors as bc
```

# Numerically solving for eigenstates and eigenvalues of an arbitrary 1D potential

Description: Obtain the energy eigenvalues $E_n$ and wavefunctions $\psi_n(r)$ for the anharmonic Morse potential (below). Note that value of the parameters correspond the hydrogen fluoride. Tabulate $E_n$ for $n=0$ to $5$, and plot the corresponding $\psi_n(r)$.

$$V = D_e [1-e^{-\alpha x}]^2$$
- **Equilibrium bond energy**:
    $$D_e=6.091\times10^{-19} \text{ J}$$
- **Equilibrium bond length**:
    $$r_0=9.109\times10^{-11} \text{ m}, \quad x=r-r_0$$
- **Force constant**:
    $$k=1.039\times10^{3} \ \text{J}\text{m}^{-2}, \quad \alpha=\sqrt{ k / 2D_e}$$


## 1️⃣ Define constants in atomic units


### Define physical constants
📝 Constants are transformed from SI units to atomic units.

```python
# Import necessary libraries
import pandas as pd
from scipy.constants import physical_constants
from IPython.display import display, HTML

# Physical constants in SI units
hbar_SI = physical_constants['Planck constant over 2 pi'][0]  # J*s
m_e_SI = physical_constants['electron mass'][0]               # kg
a_0_SI = physical_constants['Bohr radius'][0]                 # m
E_h_SI = physical_constants['Hartree energy'][0]              # J

# Physical constants in atomic units
hbar_AU = 1             # Reduced Planck constant
m_e_AU = 5.4858e-4      # Atomic mass unit (Daltons)
a_0_AU = 1              # Bohr radius
E_h_AU = 1              # Hartree energy

# Create a dictionary with the physical constants
physical_constants_data = {
    'Constant': [
        r'hbar',
        r'm_e (electron mass)',
        r'a_0',
        r'E_h'
    ],
    'Value in SI Units': [hbar_SI, m_e_SI, a_0_SI, E_h_SI],
    'SI Units': ['J·s', 'kg', 'm', 'J'], 
    'Value in Atomic Units': [hbar_AU, m_e_AU, a_0_AU, E_h_AU],
    'AU Units': ['reduced Planck constant', 'Daltons', 'Bohr radius', 'Hartree energy']
}

# Create a DataFrame
df = pd.DataFrame(physical_constants_data)
df
```

## ⚛️ Calculate $\alpha$ using the given $k$ and $D_e$

```python
import numpy as np
from scipy.constants import physical_constants

# Given values in SI units
k_SI = 1.039e3  # J/m^2
D_e_SI = 6.091e-19  # J
r_0_SI = 9.109e-11  # m

# Convert D_e and r_0 to atomic units
D_e_AU = D_e_SI / E_h_SI
r_0_AU = r_0_SI / a_0_SI

# Correct conversion of k to atomic units
# k_AU = k_SI * (a_0^2 / E_h) in atomic units
k_AU = k_SI * (a_0_SI ** 2) / E_h_SI

# Compute alpha in atomic units
alpha_AU = np.sqrt(k_AU / (2 * D_e_AU))

print(f"Equilibrium bond length = {r_0_AU:.5f} Bohr radii")
print(f"Equilibrium bond energy = {D_e_AU:.5f} Hartree")
print(f"Force constant in atomic units = {k_AU:.5f}")
print(f"Alpha in atomic units = {alpha_AU:.5f}")

```

## 2️⃣ Reduced mass, $\mu$ of H-F molecule

$$\mu = \frac{m_\text{H} m_\text{F}}{m_\text{H} + m_\text{F}}$$

```python
# Atomic mass in atomic mass units
m_H_amu = 1.00784     # atomic mass units (Daltons)
m_F_amu = 18.998403   # atomic mass units (Daltons)

# Reduced mass in atomic mass units
mu_amu = (m_H_amu * m_F_amu) / (m_H_amu + m_F_amu)   # Daltons

# Reduced mass in atomic units
m_e_amu = 5.4858e-4         # electron mass (Daltons)
mu_AU = mu_amu / m_e_amu    # reduced mass (atomic units ✔️)

# Reduced mass in atomic mass units (mu / m_e)
print(f"Reduced mass: {bc.GREEN}{mu_AU:.3e} atomic mass units{bc.ENDC}")
```

## 3️⃣ Compute harmonic frequency and $\alpha$

```python
# Compute harmonic frequency in atomic units
omega_AU = np.sqrt(k_AU / mu_AU)

# Compute alpha in atomic units
alpha_AU = np.sqrt(k_AU / (2 * D_e_AU))

print(f"Omega in atomic units = {omega_AU:.5f}")
print(f"Alpha in atomic units = {alpha_AU:.5f}")
```

## 4️⃣ Set up spatial grid

```python
# Spatial range in atomic units (typically a few Bohr radii around equilibrium)
xmin = (-2)/(a_0_AU)  # a_0
xmax = (5)/(a_0_AU)   # a_0

N = 1000  # Increase N for better resolution
x = np.linspace(xmin, xmax, N)      # Displacement from equilibrium position
dx = x[1] - x[0]

# Create a dictionary with the data, using raw strings for LaTeX expressions
physical_constants_data = {
    'Grid point/interval': [
        r'xmin',
        r'xmax',
        r'dx',
    ],
    'Value in AU Units': [xmin, xmax, dx],
    'Atomic Units': ['Bohr radii', 'Bohr radii', 'Bohr radii']
}

# Create a DataFrame
df = pd.DataFrame(physical_constants_data)
df

# Displacement from equilibrium position (already in atomic units)
```

## 4️⃣ Potential and Hamiltonian Setup


### 💙 Calculate the potential $V(x)$ at each point on the grid

```python
V_ii = D_e_AU * (1 - np.exp(-alpha_AU * x))**2   # V in Hartree units

# Set minimum value
V_floor = 1e-6  # Choose an appropriate floor value
V = np.maximum(V_ii, V_floor)
```

### 🩷 Set up the kinetic energy operator

```python
# Construct the second derivative operator (kinetic energy term)
T_coeff = (hbar_AU**2) / (2 * mu_AU * dx**2)            # Hartree units
diagonal = -2 * np.ones(N) * T_coeff
off_diagonal = (-1) * np.ones(N - 1) * T_coeff

# Assemble the kinetic energy matrix
from scipy.sparse import diags

T = diags([off_diagonal, diagonal, off_diagonal], offsets=[-1,0,1])
print(T)
```

### 🌀 Construct the Hamiltonian Matrix
Combine the kinetic and potential energy terms

```python
# Potential energy matrix (diagonal matrix)
V_matrix = diags(V, 0, format='csr')

# Hamiltonian matrix
H = T + V_matrix
```

```python
# Ensure no NaNs or infinities:
import numpy as np

if np.isnan(H.data).any() or np.isinf(H.data).any():
    print("Hamiltonian contains NaNs or Infinities. ❌")
else:
    print("Hamiltonian matrix is valid. ✅")
```

```python
# Print magnitudes of matrix elements
H_data = np.array(H.data)
max_element = np.max(np.abs(H_data))
non_zero_elements = H_data[H_data != 0]  # This extracts all non-zero elements

min_element = np.min(np.abs(non_zero_elements))
print(f"Max Hamiltonian element: {max_element:.5f} Hartrees")
print(f"Min non-zero Hamiltonian element: {min_element:.5f} Hartrees")
```

## 5️⃣ Solving the Schrodinger Equation

```python
from scipy.sparse.linalg import eigsh

# Number of eigenvalues and eigenvectors to compute
num_eigenvalues = 6

# Compute the lowest eigenvalues and corresponding eigenvectors
eigenvalues, eigenvectors = eigsh(H, k=num_eigenvalues, which='SA', tol=1e-5, maxiter=10000)

# 'which' parameter:
# 'SA' - compute the smallest algebraic eigenvalues
# 'SM' - may not work properly with sparse matrices and complex potentials
```

### 📏 Converting Eigenvalues to Physical Units

```python
# Eigenvalues are already in Joules
E_n = eigenvalues   # J
```

## 🖲️ Normalizing the Eigenfunctions
**Continuous normalization (physics)**:

**Discrete normalization (finite difference grid):**
$$ \sum_i{ | \psi_n(x) |^2 \Delta x} = 1 $$

**Normalization constant:**
$$N = \sqrt{\sum_i{ | \psi_n(x) |^2 \Delta x} }$$

```python
psi_n = []
for i in range(num_eigenvalues):
    psi = eigenvectors[:, i]

    # Normalize eigenfunctions
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    psi = psi / norm
```

### 🔖 Tabulate the Energy Eigenvalues

```python
import pandas as pd
from tabulate import tabulate

# Format the DataFrame
data = {'n': np.arange(num_eigenvalues, dtype=int), 'E_n (Hartree)': np.round(E_n, 5)}
df = pd.DataFrame(data)

# Print as a pretty table
print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))

```

## Plot the Wavefunctions

```python
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def plot_morse_potential(
    x: NDArray[np.float64],
    V: NDArray[np.float64],
    marker_every: int = 25,
) -> None:
    """
    Plot the Morse potential with point markers.

    Parameters
    ----------
    x : ndarray
        Displacement from equilibrium (Bohr radii).
    V : ndarray
        Morse potential V(x) in Hartree.
    marker_every : int, optional
        Interval for marker placement along the curve.
    """

    plt.figure(figsize=(9, 5))

    plt.plot(
        x,
        V,
        "-o",
        color="#3bb2f5",
        linewidth=2,
        markersize=3,
        markevery=marker_every,
        label=r"$V(x)$",
    )

    plt.xlabel(r"$x$ (Bohr radii)")
    plt.ylabel(r"Energy (Hartree)")
    plt.title("Morse Potential (HF)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_morse_potential(x, V)

```

```python
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def plot_morse_two_panel(
    x: NDArray[np.float64],
    psi_n: Sequence[NDArray[np.float64]],
    E_n: NDArray[np.float64],
    V: NDArray[np.float64],
    D_e: float,
) -> None:
    """
    Two-panel plot:
    (left) full Morse potential
    (right) vibrational eigenstates in the well
    """

    scale: float = 0.08 * (E_n[1] - E_n[0])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    # Left panel: full potential
    axes[0].plot(
        x,
        V,
        "-o",
        markevery=30,
        markersize=3,
        color="#007FFF",
    )
    axes[0].set_title("Morse Potential")
    axes[0].set_ylabel("Energy (Hartree)")
    axes[0].grid(alpha=0.3)

    # Right panel: bound states
    axes[1].plot(x, V, color="#007FFF", linewidth=2)

    for n, psi in enumerate(psi_n):
        axes[1].plot(x, E_n[n] + scale * psi)
        axes[1].hlines(
            E_n[n],
            x.min(),
            x.max(),
            linestyles="dotted",
            alpha=0.4,
        )

    #axes[1].set_ylim(-0.05 * D_e, 1.1 * np.max(E_n))
    axes[1].set_title("Vibrational Eigenstates")
    axes[1].grid(alpha=0.3)

    for ax in axes:
        ax.set_xlabel(r"$x$ (Bohr radii)")

    plt.tight_layout()
    plt.show()

plot_morse_two_panel(x, psi_n, E_n, V, D_e_AU)
```

```python
# All bound states must satisfy E < D_e
print(E_n / D_e_AU)

# Normalization check
for psi in psi_n:
    print(np.sum(np.abs(psi)**2) * dx)
```
