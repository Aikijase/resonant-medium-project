import numpy as np
from dataclasses import dataclass

C_KMS = 299792.458  # km/s

@dataclass
class LCDM:
    H0: float = 70.0
    Omega_m: float = 0.3
    Omega_L: float = None  # if None, use flatness: 1-Omega_m

    def E(self, z):
        Om = self.Omega_m
        Ol = (1.0 - Om) if self.Omega_L is None else self.Omega_L
        return np.sqrt(Om*(1+z)**3 + Ol)

    def comoving_distance(self, z):
        z = np.atleast_1d(z).astype(float)
        # simple trapz integral of 1/E(z)
        out = []
        for zmax in z:
            zz = np.linspace(0.0, zmax, 4096)
            out.append(np.trapz(1.0/self.E(zz), zz))
        return (C_KMS/self.H0) * np.array(out)  # Mpc

    def luminosity_distance(self, z):
        z = np.atleast_1d(z).astype(float)
        chi = self.comoving_distance(z)
        return (1.0 + z) * chi  # Mpc

    def distance_modulus(self, z):
        dl_mpc = self.luminosity_distance(z)
        return 5.0*np.log10(dl_mpc*1e6) - 5.0  # mag
