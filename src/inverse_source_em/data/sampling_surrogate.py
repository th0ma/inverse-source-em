import numpy as np

def sample_sources(N, R=1.0, rho_min=0.05, rho_max=0.95):
    """
    Sample N source positions uniformly in the annulus:
        rho ∈ [rho_min*R, rho_max*R]
        phi ∈ [0, 2π)
    rho is sampled with sqrt() for uniform area density.
    """
    r2_min = (rho_min / R)**2
    r2_max = (rho_max / R)**2

    rho = np.sqrt(np.random.uniform(r2_min, r2_max, size=N)) * R
    phi = np.random.uniform(0.0, 2*np.pi, size=N)
    return rho, phi


def sample_angles(M):
    """Uniform sampling of M observation angles in [0, 2π)."""
    return np.linspace(0.0, 2*np.pi, M, endpoint=False)
