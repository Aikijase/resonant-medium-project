import numpy as np
def _H(z, p):
    A,f,phi,gamma = p["A"],p["f"],p["phi"],p["gamma"]
    return np.clip(np.exp(-gamma*z)*(1.0 + A*np.cos(2*np.pi*f*z+phi)), 1e-9, None)
def phi_proxy(z, p):
    H = _H(z, p)
    return (1.0/H**2)*np.exp(-z)
def cl_tg_proxy(p, bins):
    ell = np.arange(10,100)
    out=[]
    for (zc,w) in bins:
        z = np.linspace(max(0.0,zc-w), zc+w, 64)
        amp = phi_proxy(z,p).mean()
        Cl  = amp*np.exp(-ell/50.0)
        out.append({"z_mean":zc,"z_halfwidth":w,"ell":ell.tolist(),"Cl":Cl.tolist()})
    return out
