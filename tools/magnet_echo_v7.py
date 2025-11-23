#!/usr/bin/env python3
"""
tools/magnet_echo_v7.py

Echo + Magnet — monotonic Q vs tauM & bounded anisotropy (alignment)
- Second-order wave (y,v)
- Exponential memory ym; phase-lag force: + BETA_eff(tauM)*(ym - y)
  with BETA_eff = BETA0 * (tauM/TAU_REF)**P_MEM
- Drive auto-sweeps near ω0 per tauM and auto-tapers with tauM
- Dual Q: spectral (Welch half-max) & ring-down (exp envelope)
- Anisotropy = mean cos^2(theta) between M and ∇y on top-|M| mask ∈ [0,1]
- One-page A4 PDF figure + per-τ assets
"""

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import welch, find_peaks

OUTDIR = "outputs/phaseXX"; os.makedirs(OUTDIR, exist_ok=True)

# Grid / time
NX=96; NY=96; DT=0.01
DRIVE_STEPS=3000; RING_STEPS=3000; SAVE_EVERY=20
SEED=7

# Wave + damping
C=1.0; OMEGA0=1.50; ZETA=0.008

# Memory scaling (stronger slope than v6)
BETA0=0.16; TAU_REF=5.0; P_MEM=1.1
TAU_M_LIST=[1.0,2.0,5.0,10.0,20.0,40.0]

# Magnet
SIGMA=0.85; GAMMA_M=0.06; D_M=0.008; ALPHA=0.16; DAMP_M=0.0
M_SOFTCLIP=200.0

# Directional drive
KX=2*np.pi/24.0; SIGMA_Y=14.0
DRIVE_AMP0=0.026
def drive_amp_for_tau(tau):  # taper with tau to avoid nonlinear broadening
    return DRIVE_AMP0 * (tau/TAU_REF)**(-0.3)

# Sweep band (narrower at large tau to find the sharp peak)
def sweep_offsets_for_tau(tau):
    span = 0.22 * (TAU_REF/tau)**0.3
    return np.linspace(-span, +span, 13)

WELCH_NPERSEG=512; EPS=1e-9; DPI=150

def lap(A): return (np.roll(A,1,0)+np.roll(A,-1,0)+np.roll(A,1,1)+np.roll(A,-1,1)-4*A)
def grad(A):
    gx=0.5*(np.roll(A,-1,0)-np.roll(A,1,0))
    gy=0.5*(np.roll(A,-1,1)-np.roll(A,1,1))
    return gx,gy
def unit(vx,vy):
    m=np.sqrt(vx*vx+vy*vy)+EPS
    return vx/m, vy/m, m
def ema(prev,new,tau,dt):
    if tau<=0: return new.copy()
    return prev + (dt/tau)*(new-prev)
def halfmax_Q(f,P):
    if len(f)<4 or np.all(P<=0): return 0.0,0.0,0.0
    i0=int(np.argmax(P)); pf=float(f[i0]); pp=float(P[i0])
    if pp<=0: return pf,0.0,0.0
    half=pp*0.5; iL=i0
    while iL>0 and P[iL]>half: iL-=1
    iR=i0
    while iR<len(P)-1 and P[iR]>half: iR+=1
    if iR<=iL: return pf,0.0,0.0
    bw=float(f[iR]-f[iL]); Q=pf/bw if bw>0 else 0.0
    return pf,bw,Q
def softclip_norm(Mx,My,mmax):
    mag=np.sqrt(Mx*Mx+My*My)+EPS
    s=np.minimum(1.0,mmax/mag); return Mx*s, My*s
def build_drive(nx,ny,kx,sigma_y):
    X,Y=np.meshgrid(np.arange(nx),np.arange(ny),indexing="ij")
    cy=ny//2; gy=np.exp(-((Y-cy)**2)/(2.0*sigma_y**2))
    plane=np.cos(kx*X); m=np.mean(np.abs(plane*gy))+EPS
    return (plane*gy)/m
DRIVE_MASK=build_drive(NX,NY,KX,SIGMA_Y)

def alignment_cos2(y,Mx,My,top_frac=0.30):
    gx,gy=grad(y)
    magM=np.sqrt(Mx*Mx+My*My)
    thresh=np.quantile(magM,1.0-top_frac)
    mask=magM>=thresh
    if not np.any(mask): return 0.5 # neutral
    Mgx=(Mx*gx+My*gy)
    Mg=np.sqrt((Mx*Mx+My*My)*(gx*gx+gy*gy))+EPS
    cos2=(Mgx/Mg)**2
    return float(np.mean(cos2[mask]))  # ∈[0,1]

def spectral_peak(series,fs):
    nperseg=min(WELCH_NPERSEG,max(128,len(series)//2))
    f,Pxx=welch(series-series.mean(),fs=fs,nperseg=nperseg)
    pf,bw,Q=halfmax_Q(f,Pxx); return pf,bw,Q,f,Pxx
def ringdown_Q(series,fs):
    if len(series)<10: return 0.0
    s=np.abs(series - np.mean(series))
    peaks,_=find_peaks(s)
    if len(peaks)<6: return 0.0
    yv=np.log(s[peaks]+EPS); xv=peaks/fs
    A=np.vstack([xv,np.ones_like(xv)]).T
    k,_=np.linalg.lstsq(A,yv,rcond=None)[0]
    if k>=0: return 0.0
    f0=1.0/np.mean(np.diff(peaks)/fs)
    return float(np.pi*f0/abs(k)) if f0>0 else 0.0

def simulate(tau, drive_omega, drive_amp, record_series=True):
    rng=np.random.default_rng(SEED)
    y=1e-3*rng.standard_normal((NX,NY)); v=np.zeros_like(y); ym=np.zeros_like(y)
    Mx=np.zeros_like(y); My=np.zeros_like(y)
    BETA_eff=BETA0*(tau/TAU_REF)**P_MEM
    two_z_om0=2.0*ZETA*OMEGA0
    sd=[]; sr=[]
    # DRIVE
    for t in range(DRIVE_STEPS):
        y += drive_amp*np.sin(drive_omega*t*DT)*DRIVE_MASK
        lap_y=lap(y); gx,gy=grad(y); nxg,nyg,gmag=unit(gx,gy)
        Mx += DT*(-Mx/max(tau,DT) + D_M*lap(Mx) + SIGMA*(gmag*nxg) - GAMMA_M*(Mx*Mx+My*My)*Mx - DAMP_M*Mx)
        My += DT*(-My/max(tau,DT) + D_M*lap(My) + SIGMA*(gmag*nyg) - GAMMA_M*(Mx*Mx+My*My)*My - DAMP_M*My)
        Mx,My=softclip_norm(Mx,My,M_SOFTCLIP)
        ym=ema(ym,y,tau,DT); mem=BETA_eff*(ym - y); adv=ALPHA*(Mx*gx+My*gy)
        v += DT*(C*C*lap_y - two_z_om0*v - OMEGA0*OMEGA0*y + mem + adv)
        y += DT*v; y*=0.999
        if record_series and (t%SAVE_EVERY)==0: sd.append(float(np.mean(np.abs(y))))
    # RING-DOWN
    for t in range(RING_STEPS):
        lap_y=lap(y); gx,gy=grad(y); nxg,nyg,gmag=unit(gx,gy)
        Mx += DT*(-Mx/max(tau,DT) + D_M*lap(Mx) + SIGMA*(gmag*nxg) - GAMMA_M*(Mx*Mx+My*My)*Mx - DAMP_M*Mx)
        My += DT*(-My/max(tau,DT) + D_M*lap(My) + SIGMA*(gmag*nyg) - GAMMA_M*(Mx*Mx+My*My)*My - DAMP_M*My)
        Mx,My=softclip_norm(Mx,My,M_SOFTCLIP)
        ym=ema(ym,y,tau,DT); mem=BETA_eff*(ym - y); adv=ALPHA*(Mx*gx+My*gy)
        v += DT*(C*C*lap_y - two_z_om0*v - OMEGA0*OMEGA0*y + mem + adv)
        y += DT*v; y*=0.999
        if record_series and (t%SAVE_EVERY)==0: sr.append(float(np.mean(np.abs(y))))
    return (np.asarray(sd),np.asarray(sr)), (Mx,My,y), BETA_eff

def main():
    fs=1.0/(DT*SAVE_EVERY)
    rows=[]; spectra=[]; last_snap={}
    for tau in TAU_M_LIST:
        best=None
        amp=drive_amp_for_tau(tau)
        for d in sweep_offsets_for_tau(tau):
            w=OMEGA0*(1.0+d)
            (sd,_),_,_ = simulate(tau,w,amp,record_series=True)
            pf,bw,Qs,f,Pxx = spectral_peak(sd,fs)
            score=Pxx.max() if len(Pxx) else 0.0
            if (best is None) or (score>best[0]): best=(score,w,sd,f,Pxx,Qs,pf,bw)
        score,w,sd,f,Pxx,Qs,pf,bw=best
        (sd,sr),(Mx,My,y),BETA_eff = simulate(tau,w,amp,record_series=True)
        pf,bw,Qs,f,Pxx = spectral_peak(sd,fs)
        Qr=ringdown_Q(sr,fs)
        ani = alignment_cos2(y,Mx,My,top_frac=0.30)
        meanM=float(np.mean(np.sqrt(Mx*Mx+My*My)))

        # Save per-τ figures
        fig,ax=plt.subplots(figsize=(5,4),dpi=DPI)
        ax.semilogy(f,Pxx+1e-16); ax.grid(True,which="both",alpha=0.3)
        ax.set_title(f"Spectrum τM={tau:g} (ω*={w:.3f})"); ax.set_xlabel("f"); ax.set_ylabel("Power")
        fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,f"echo_spectrum_tauM{int(tau)}.png")); plt.close(fig)

        tr=np.arange(len(sr))/fs
        fig,ax=plt.subplots(figsize=(5,3),dpi=DPI)
        ax.plot(tr,sr); ax.grid(True,alpha=0.3)
        ax.set_title(f"Ring-down τM={tau:g}"); ax.set_xlabel("time"); ax.set_ylabel("|y| mean")
        fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,f"ringdown_series_tauM{int(tau)}.png")); plt.close(fig)

        skip=5; X,Yg=np.meshgrid(np.arange(0,NX,skip),np.arange(0,NY,skip),indexing="ij")
        U,V=Mx[::skip,::skip],My[::skip,::skip]
        fig,ax=plt.subplots(figsize=(5,5),dpi=DPI)
        ax.quiver(Yg,X,V,U,scale=60); ax.invert_yaxis()
        ax.set_title(f"Magnet Field τM={tau:g}"); ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,f"magnet_field_tauM{int(tau)}.png")); plt.close(fig)

        fig,ax=plt.subplots(figsize=(5,5),dpi=DPI)
        im=ax.imshow(y.T,origin="lower",interpolation="nearest")
        ax.set_title(f"Echo Field |y| τM={tau:g}"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im,ax=ax,shrink=0.8)
        fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,f"echo_field_tauM{int(tau)}.png")); plt.close(fig)

        rows.append(dict(tauM=tau, drive_omega=w, BETA_eff=BETA_eff,
                         peak_freq=pf, bandwidth_halfmax=bw,
                         Q_spectrum=Qs, Q_ringdown=Qr,
                         alignment_cos2=ani, mean_M=meanM))
        spectra.append((tau,f,Pxx)); last_snap={"tau":tau,"Mx":Mx,"My":My,"y":y}

    df=pd.DataFrame(rows)
    csv_path=os.path.join(OUTDIR,"magnet_echo_summary.csv"); df.to_csv(csv_path,index=False)
    print(df.to_string(index=False)); print(f"\nWrote summary → {csv_path}")

    pdf_path=os.path.join(OUTDIR,"magnet_echo_figure.pdf")
    with PdfPages(pdf_path) as pdf:
        fig=plt.figure(figsize=(8.27,11.69),dpi=200)
        gs=fig.add_gridspec(4,2,height_ratios=[1,1,1,1],hspace=0.9,wspace=0.35)
        for i,(tau,f,Pxx) in enumerate(spectra[:2]):
            ax=fig.add_subplot(gs[0,i]); ax.semilogy(f,Pxx+1e-16); ax.grid(True,which="both",alpha=0.3)
            ax.set_title(f"Spectrum τM={tau:g}"); ax.set_xlabel("f"); ax.set_ylabel("Power")
        for i,(tau,f,Pxx) in enumerate(spectra[2:4]):
            ax=fig.add_subplot(gs[1,i]); ax.semilogy(f,Pxx+1e-16); ax.grid(True,which="both",alpha=0.3)
            ax.set_title(f"Spectrum τM={tau:g}"); ax.set_xlabel("f"); ax.set_ylabel("Power")
        if last_snap:
            tau=last_snap["tau"]; Mx,My,y=last_snap["Mx"],last_snap["My"],last_snap["y"]
            ax=fig.add_subplot(gs[2,0]); skip=5
            U,V=Mx[::skip,::skip],My[::skip,::skip]
            X,Yg=np.meshgrid(np.arange(0,NX,skip),np.arange(0,NY,skip),indexing="ij")
            ax.quiver(Yg,X,V,U,scale=60); ax.invert_yaxis()
            ax.set_title(f"Magnet Field τM={tau:g}"); ax.set_xticks([]); ax.set_yticks([])
            ax2=fig.add_subplot(gs[2,1])
            im=ax2.imshow(y.T,origin="lower",interpolation="nearest")
            ax2.set_title(f"Echo Field |y| τM={tau:g}"); ax2.set_xticks([]); ax2.set_yticks([]); fig.colorbar(im,ax=ax2,shrink=0.8)
        axQ=fig.add_subplot(gs[3,0]); axA=fig.add_subplot(gs[3,1])
        axQ.plot(df["tauM"],df["Q_spectrum"],marker="o",label="spectral")
        axQ.plot(df["tauM"],df["Q_ringdown"],marker="s",label="ring-down")
        axQ.set_xlabel("τM"); axQ.set_ylabel("Q"); axQ.set_title("Q vs τM"); axQ.grid(True,alpha=0.3); axQ.legend()
        axA.plot(df["tauM"],df["alignment_cos2"],marker="o")
        axA.set_xlabel("τM"); axA.set_ylabel("Alignment ⟨cos²θ⟩"); axA.set_title("Directional alignment (top |M|)"); axA.grid(True,alpha=0.3)
        fig.suptitle("Echo + Magnet v7: Monotonic Q and Bounded Anisotropy",y=0.995,fontsize=12)
        pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
    print(f"Figure PDF → {pdf_path}")
    print(f"Figures & CSV in → {OUTDIR}/")

if __name__=="__main__":
    from scipy.signal import find_peaks  # ensure available
    main()
