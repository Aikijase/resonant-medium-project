import numpy as np
from dataclasses import dataclass

C_KMS = 299792.458  # km/s

@dataclass
class ResonantDE:
    H0: float = 70.0
    Omega_m: float = 0.3
    # Resonance in the DE sector:
    A: float = 0.05        # amplitude in w(a) = -1 + A * sin(f ln a + phi)
    f: float = 1.0         # frequency in ln a
    phi: float = 0.0       # phase

    def E(self, z):
        a = 1.0/(1.0+np.atleast_1d(z))
        Om = self.Omega_m
        # DE density scaling:
        # ln rho_de(a) - ln rho_de(1) = 3*A/f * [cos(f ln a + phi) - cos(phi)]
        f = self.f if abs(self.f) > 1e-6 else 1e-6
        osc = np.exp(3.0*self.A/f * (np.cos(f*np.log(a) + self.phi) - np.cos(self.phi)))
        Ode = 1.0 - Om
        E2 = Om*a**-3 + Ode*osc
        return np.sqrt(np.maximum(E2, 1e-12))

    def comoving_distance(self, z):
        z = np.atleast_1d(z).astype(float)
        out = []
        for zmax in z:
            zz = np.linspace(0.0, zmax, 4096)
            out.append(np.trapz(1.0/self.E(zz), zz))
        return (C_KMS/self.H0) * np.array(out)

    def luminosity_distance(self, z):
        z = np.atleast_1d(z).astype(float)
        chi = self.comoving_distance(z)
        return (1.0 + z)*chi

    def distance_modulus(self, z):
        dl_mpc = self.luminosity_distance(z)
        return 5.0*np.log10(dl_mpc*1e6) - 5.0
