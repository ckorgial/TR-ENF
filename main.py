#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author Christos Korgialas

"""
ENF-WHU — 100 Hz Trust-Region ENF Detection (GLRT/LS-LRT)

This version produces:
  1) OUT_DIR/roc_trenf_5s.png           # ROC curve for TR-ENF, 5s
  2) OUT_DIR/det_curves_trenf_5_10s.png # DET curves for TR-ENF, 5–10s

Other outputs:
  out_dir/per_segment_tr.csv
  out_dir/summary_tr.csv
  out_dir/tr_convergence_perseg.csv

Install & run:
  pip install numpy scipy soundfile pandas matplotlib
  python main.py
"""

from __future__ import annotations
import os, sys, math, time, random, logging, warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import firwin2, lfilter, resample_poly
from scipy.optimize import minimize_scalar
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

# === import Trust-Region optimizer from external module ===
from tr_optimizer import build_problem, trust_region

# ======================
# Configuration
# ======================
BASE_DIR      = "./recordings"
OUT_DIR       = "./results"
FS_TARGET     = 400
SEG_DURS      = [5, 6, 7, 8, 9, 10]
SEED          = 1234
MAX_TR_ITERS  = 100
K_SEGMENTS    = 5
ENERGY_MIN    = 1e-8
PRINT_SEGMENT = True

GLRT_MODE     = "localmax"
LOCAL_HALF_HZ = 0.4
BOOTSTRAP_B   = 200

warnings.filterwarnings("ignore", message="invalid value encountered in multiply",
                        module=r"scipy\.optimize\._lsq\.trf_linear")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s")
logger = logging.getLogger("TR-DET-100Hz")

rng = np.random.default_rng(SEED)
random.seed(SEED)

# ======================
# Bandpass filter (100 Hz peak at fs=400 Hz)
# ======================
F2 = np.array([0.00,0.40,0.495,0.499,0.500,0.501,0.505,0.60,0.80,1.00],dtype=float)
M2 = np.array([0.00,0.00,0.00,0.20,1.00,0.20,0.00,0.00,0.00,0.00],dtype=float)
BPF_TAPS=1023

def _normalize_bpf(bpf: np.ndarray) -> np.ndarray:
    nfft=8192
    bpff=np.abs(np.fft.rfft(bpf,n=nfft))
    scalar=np.max(bpff) if bpff.size else 1.0
    return bpf/max(scalar,1e-12)

BPF=_normalize_bpf(firwin2(BPF_TAPS,F2,M2))

# ======================
# Utilities
# ======================
@dataclass
class FileEntry:
    path: str
    label: int  # 1=H1, 0=H0

def collect_files(root:str)->list[FileEntry]:
    exts=(".wav",".flac",".m4a")
    out=[]
    for sub,lab in [("H1",1),("H0",0)]:
        d=os.path.join(root,sub)
        if not os.path.isdir(d):
            logger.warning(f"Missing folder: {d}")
            continue
        for r,_,fns in os.walk(d):
            for f in sorted(fns):
                if f.lower().endswith(exts):
                    out.append(FileEntry(os.path.join(r,f),lab))
    return out

def load_audio_mono_norm(path:str)->tuple[np.ndarray,float]:
    x,Fs=sf.read(path,always_2d=False)
    if x.ndim>1: x=x[:,0]
    x=np.asarray(x,float)
    m=np.max(np.abs(x)) if x.size>0 else 0.0
    if m>0: x=x/m
    return x,float(Fs)

def ds_and_filter(x:np.ndarray,Fs_in:float)->tuple[np.ndarray,float]:
    if abs(Fs_in-FS_TARGET)>1e-6:
        from math import gcd
        g=gcd(int(round(FS_TARGET)),int(round(Fs_in)))
        up=int(round(FS_TARGET))//g; down=int(round(Fs_in))//g
        x=resample_poly(x,up,down); Fs=FS_TARGET
    else: Fs=Fs_in
    x=x-float(np.mean(x))
    xf=lfilter(BPF,[1.0],x)
    return xf.astype(float),Fs

def energy_ok(x:np.ndarray,min_norm=ENERGY_MIN)->bool:
    E=float(np.linalg.norm(x))
    return np.isfinite(E) and (E>=min_norm)

def rfft_peak_freq(x:np.ndarray,Fs:float)->float:
    N=x.size
    nfft=1<<max(18,int(math.ceil(math.log2(max(N,1))))+2)
    X=np.abs(np.fft.rfft(x,n=nfft))
    k=int(np.argmax(X))
    return (k*Fs)/nfft

# ======================
# GLRT / LS-LRT
# ======================
def ls_lrt_stat(xf:np.ndarray,fc:float,T:float)->float:
    N=xf.size
    k=np.arange(N,dtype=float)
    c=np.cos(2*np.pi*T*fc*k); s=np.sin(2*np.pi*T*fc*k)
    num=(xf@c)**2+(xf@s)**2
    den=float(np.linalg.norm(xf)**2)+1e-12
    return float((2.0/N)*num/den)

def tr_glrt_localmax(xf: np.ndarray, fc_tr: float, T: float, half_width_hz: float = 0.2) -> tuple[float, float]:
    a, b = fc_tr - half_width_hz, fc_tr + half_width_hz
    res = minimize_scalar(lambda f: -ls_lrt_stat(xf, f, T),
                          bounds=(a, b), method="bounded")
    f_best = float(res.x if res.success else fc_tr)
    s_best = ls_lrt_stat(xf, f_best, T)
    # clip to [99.8, 100.2] Hz for stability
    f_best = float(np.clip(f_best, 99.8, 100.2))
    return f_best, s_best


# ======================
# Metrics helpers
# ======================
def roc_curve_step(scores:np.ndarray, labels:np.ndarray):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    P = int(np.sum(labels == 1)); N = int(np.sum(labels == 0))
    if P == 0 or N == 0:
        return np.array([0.0,1.0]), np.array([0.0,1.0]), 0.5
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y == 1).astype(float)
    fp = np.cumsum(y == 0).astype(float)
    tpr = tp / P; fpr = fp / N
    tpr = np.r_[0.0, tpr, 1.0]; fpr = np.r_[0.0, fpr, 1.0]
    order2 = np.argsort(fpr)
    fpr_sorted, tpr_sorted = fpr[order2], tpr[order2]
    auc_val = float(np.trapz(tpr_sorted, fpr_sorted))
    return fpr_sorted, tpr_sorted, auc_val

def bootstrap_acc_std_at_threshold(scores,labels,thr,B=200,seed=1):
    rng_local=np.random.default_rng(SEED+seed); n=len(scores); ids=np.arange(n)
    accs=[]
    for _ in range(B):
        idx=rng_local.choice(ids,size=n,replace=True)
        yhat=(scores[idx]>=thr).astype(int)
        accs.append(float(np.mean(yhat==labels[idx])))
    accs=np.array(accs)
    return float(accs.mean()),float(accs.std(ddof=1))

# ======================
# Plotting helpers
# ======================
def smooth_curve(x,y,num=1000):
    x=np.asarray(x,float); y=np.asarray(y,float)
    order=np.argsort(x); x,y=x[order],y[order]
    uniq_x,idx=np.unique(x,return_index=True); uniq_y=y[idx]
    if uniq_x.size<2: return uniq_x,uniq_y
    f=PchipInterpolator(uniq_x,uniq_y)
    x_new=np.linspace(uniq_x.min(),uniq_x.max(),num)
    return x_new,f(x_new)

def plot_trenf_roc_5s(summary,out_png):
    row=summary[summary["duration_s"]==5].iloc[0]
    fpr,tpr,auc=row["fpr"],row["tpr"],float(row["auc"])
    fpr_s,tpr_s=smooth_curve(fpr,tpr)
    plt.figure(figsize=(6.5,6.5),dpi=150)
    plt.plot(fpr_s*100,tpr_s*100,color="darkorange",lw=3,
             label=f"TR-ENF, 5s (AUC={auc*100:.1f}%)")
    plt.plot([0,100],[0,100],'k--',lw=1.5)
    plt.xlabel("False Positive Rate (%)"); plt.ylabel("True Positive Rate (%)")
    plt.axis("square"); plt.xlim([0,100]); plt.ylim([0,100])
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_trenf_det_curves(summary,out_png):
    plt.figure(figsize=(6.5,6.5),dpi=150)
    colors={5:"orange",6:"blue",7:"green",8:"olive",9:"navy",10:"darkorange"}
    for d in [5,6,7,8,9,10]:
        row=summary[summary["duration_s"]==d].iloc[0]
        fpr,tpr=row["fpr"],row["tpr"]; fnr=1-tpr
        fpr_s,fnr_s=smooth_curve(fpr*100,fnr*100)
        diff=np.abs(fpr-fnr); idx=np.argmin(diff)
        eer_raw=(fpr[idx]+fnr[idx])/2*100
        diff_s=np.abs(fpr_s-fnr_s); idx_s=np.argmin(diff_s)
        eer_s=(fpr_s[idx_s]+fnr_s[idx_s])/2
        print(f"[{d}s] Raw EER={eer_raw:.2f}%, Smoothed EER={eer_s:.2f}%")
        plt.plot(fpr_s,fnr_s,lw=2.5,color=colors[d],
                 label=f"TR-ENF, {d}s (EER={eer_s:.2f}%)")
        plt.scatter([fpr_s[idx_s]],[fnr_s[idx_s]],s=40,color=colors[d],zorder=5)
    plt.plot([0,100],[0,100],'k--',lw=1.5,label="EER line")
    plt.xlabel("False Positive Rate (%)"); plt.ylabel("False Negative Rate (%)")
    plt.axis("square"); plt.xlim([0,100]); plt.ylim([0,100])
    plt.legend(loc="upper right"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png); plt.close()

# ======================
# Processing
# ======================
def process_duration(files,dsec,tr_converg_rows):
    print(f"\n========== Duration {dsec}s ==========")
    rows,file_scores,file_labels=[],[],[]
    for fe in files:
        x,Fs0=load_audio_mono_norm(fe.path)
        need=int(dsec*Fs0)
        if x.size<=need+10:
            if PRINT_SEGMENT: print(f"[skip] {os.path.basename(fe.path)} (too short)")
            continue
        p_list,used_segments=[],0
        for k in range(K_SEGMENTS):
            start=rng.integers(0,x.size-need); seg=x[start:start+need]
            xf,Fs=ds_and_filter(seg,Fs0)
            if xf.size<100 or not energy_ok(xf):
                if PRINT_SEGMENT: print(f"  (seg {k+1}/{K_SEGMENTS}) low energy → skip")
                continue
            T=1.0/Fs; fc_fft=rfft_peak_freq(xf,Fs)
            df = 1.0 / (len(x) * T)  # frequency resolution
            seeds = [
                np.array([fc_fft, 0.1, 0.1]),  # FFT guided
                np.array([fc_fft + 0.01, 0.1, 0.1]),
                np.array([fc_fft - 0.01, 0.1, 0.1]),
                np.array([fc_fft + 0.02, 0.1, 0.1]),
                np.array([fc_fft - 0.02, 0.1, 0.1]),
                np.array([100.0, 0.1, 0.1]),  # nominal anchors
                np.array([100.2, 0.1, 0.1]),
                np.array([99.8, 0.1, 0.1]),
            ]
            J,gJ,HJ=build_problem(xf,T)
            best_mu,best_J,best_info=None,np.inf,None
            for mu0 in seeds:
                try:
                    sol,info=trust_region(J,gJ,HJ,mu0,max_iters=MAX_TR_ITERS)
                    final_J=float(J(sol))
                    if PRINT_SEGMENT:
                        print(f"  (seg {k+1}/{K_SEGMENTS}) init fc={mu0[0]:6.2f} Hz → "
                              f"fc*={sol[0]:7.4f} Hz, J={final_J:.3e}, iters={info['iters']}, "
                              f"time={info['time_sec']:.3f}s")
                    if final_J<best_J:
                        best_J,best_mu,best_info=final_J,sol,info
                except Exception as e:
                    if PRINT_SEGMENT: print(f"  (seg {k+1}/{K_SEGMENTS}) TR failed: {e}")
                    continue
            if best_mu is None:
                if PRINT_SEGMENT: print(f"  (seg {k+1}/{K_SEGMENTS}) no valid solution → skip")
                continue
            tr_converg_rows.append({
                "filepath":fe.path,"label":fe.label,"duration_s":dsec,
                "segment_index":k+1,"iters":best_info["iters"],
                "time_sec":best_info["time_sec"],"J_final":best_info["J_final"]
            })
            fc_opt=best_mu[0]
            if GLRT_MODE=="localmax":
                fc_ref,p_tr=tr_glrt_localmax(xf,fc_opt,T,LOCAL_HALF_HZ)
                if PRINT_SEGMENT: print(f"      → GLRT local max: fc_ref={fc_ref:7.4f} Hz, stat={p_tr:.6f}")
            else:
                fc_ref,p_tr=fc_opt,ls_lrt_stat(xf,fc_opt,T)
                if PRINT_SEGMENT: print(f"      → GLRT at fc_TR: fc={fc_ref:7.4f} Hz, stat={p_tr:.6f}")
            rows.append({
                "filepath":fe.path,"label":fe.label,"duration_s":dsec,
                "segment_index":k+1,"fc_tr":fc_opt,"p_tr_lslrt":p_tr
            })
            p_list.append(p_tr); used_segments+=1
        if used_segments==0:
            if PRINT_SEGMENT: print(f"[skip] {os.path.basename(fe.path)} (no valid segments)")
            continue
        p_file_stat=float(np.median(p_list))
        file_scores.append(p_file_stat); file_labels.append(fe.label)
        print(f"[file] {os.path.basename(fe.path)} label={fe.label} "
              f"segments_used={used_segments}/{K_SEGMENTS} "
              f"p_file(median)={p_file_stat:.6g}")
    y,P=np.array(file_labels,int),np.array(file_scores,float)
    if y.size==0:
        return rows,{"duration_s":dsec,"files_used":0,
                     "mode_median_ACC":np.nan,"acc_boot_std":np.nan,
                     "fpr":[],"tpr":[],"auc":np.nan}
    thr=float(np.median(P)); pred=(P>=thr).astype(int)
    acc=np.mean(pred==y)
    acc_boot,acc_std=bootstrap_acc_std_at_threshold(P,y,thr,BOOTSTRAP_B,dsec)
    fpr,tpr,auc_val=roc_curve_step(P,y)
    print(f"--- Duration {dsec}s summary (files={y.size}) ---")
    print(f"  ACC={acc*100:.2f}% (boot μ={acc_boot*100:.2f}%, σ={acc_std*100:.2f}%) "
          f"AUC={auc_val*100:.2f}%")
    return rows,{"duration_s":dsec,"files_used":int(y.size),
                 "mode_median_ACC":acc,"acc_boot_std":acc_std,
                 "fpr":fpr,"tpr":tpr,"auc":auc_val}

# ======================
# Main
# ======================
def main():
    t0=time.time(); os.makedirs(OUT_DIR,exist_ok=True)
    files=collect_files(BASE_DIR)
    if not files: sys.exit("No files found")
    all_rows,summaries,tr_converg_rows=[],[],[]
    for d in SEG_DURS:
        rows,summ=process_duration(files,d,tr_converg_rows)
        all_rows.extend(rows); summaries.append(summ)
    pd.DataFrame(all_rows).to_csv(os.path.join(OUT_DIR,"per_segment_tr.csv"),index=False)
    pd.DataFrame(summaries).to_csv(os.path.join(OUT_DIR,"summary_tr.csv"),index=False)
    pd.DataFrame(tr_converg_rows).to_csv(os.path.join(OUT_DIR,"tr_convergence_perseg.csv"),index=False)
    df_sum=pd.DataFrame(summaries)
    plot_trenf_roc_5s(df_sum,os.path.join(OUT_DIR,"roc_trenf_5s.png"))
    plot_trenf_det_curves(df_sum,os.path.join(OUT_DIR,"det_curves_trenf_5_10s.png"))
    print(f"Done in {time.time()-t0:.1f}s, outputs in {OUT_DIR}")

if __name__=="__main__":
    main()
