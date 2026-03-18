#!/usr/bin/env python3
"""Standalone utility to send joint position commands to Unitree G1-29.

Usage examples:
    # All arm joints go to zero (home)
    python g1_joint_cmd.py go_home

    # Release motor control (arms hang naturally)
    python g1_joint_cmd.py release

    # Go home then release
    python g1_joint_cmd.py go_home --release

    # Send custom 14-dim arm target (L_SP L_SR L_SY L_E L_WR L_WP L_WY  R_SP R_SR R_SY R_E R_WR R_WP R_WY)
    python g1_joint_cmd.py set -- 0.5 0 0 0.3 0 0 0  0.5 0 0 0.3 0 0 0

    # Move through waypoints defined in a file (one line per waypoint, 14 floats each)
    python g1_joint_cmd.py waypoints --file my_waypoints.txt

    # Print current joint angles and exit
    python g1_joint_cmd.py read
"""
import numpy as np
import time
import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import logging_mp
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

from teleop.robot_control.robot_arm import (
    G1_29_ArmController,
    G1_29_JointIndex,
    G1_29_JointArmIndex,
    G1_29_Num_Motors,
    G1_29_LowState,
    DataBuffer,
)

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState

ARM_JOINT_NAMES = [
    "L_ShoulderPitch", "L_ShoulderRoll", "L_ShoulderYaw", "L_Elbow",
    "L_WristRoll", "L_WristPitch", "L_WristYaw",
    "R_ShoulderPitch", "R_ShoulderRoll", "R_ShoulderYaw", "R_Elbow",
    "R_WristRoll", "R_WristPitch", "R_WristYaw",
]


def read_joint_state_readonly():
    """Read-only: subscribe to joint state without publishing any commands."""
    ChannelFactoryInitialize(0)
    sub = ChannelSubscriber("rt/lowstate", hg_LowState)
    sub.Init()

    msg = None
    for _ in range(50):
        msg = sub.Read()
        if msg is not None:
            break
        time.sleep(0.1)

    if msg is None:
        logger_mp.error("Failed to receive joint state from DDS.")
        return

    print("\n===== All body joints =====")
    for m in G1_29_JointIndex:
        q = msg.motor_state[m.value].q
        print(f"  [{m.value:2d}] {m.name:30s} = {q:+.4f} rad  ({np.degrees(q):+.1f}°)")

    print("\n===== Arm joints (14-dim vector) =====")
    for i, jid in enumerate(G1_29_JointArmIndex):
        q = msg.motor_state[jid.value].q
        print(f"  [{i:2d}] {ARM_JOINT_NAMES[i]:20s} = {q:+.4f} rad  ({np.degrees(q):+.1f}°)")
    print()


def print_all_joints(ctrl):
    all_q = ctrl.get_current_motor_q()
    print("\n===== All body joints =====")
    for m in G1_29_JointIndex:
        print(f"  [{m.value:2d}] {m.name:30s} = {all_q[m.value]:+.4f} rad  ({np.degrees(all_q[m.value]):+.1f}°)")
    print()
    arm_q = ctrl.get_current_dual_arm_q()
    print("===== Arm joints (14-dim vector) =====")
    for i, name in enumerate(ARM_JOINT_NAMES):
        print(f"  [{i:2d}] {name:20s} = {arm_q[i]:+.4f} rad  ({np.degrees(arm_q[i]):+.1f}°)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Send joint commands to Unitree G1-29",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- read ---
    sub.add_parser("read", help="Print current joint angles")

    # --- go_home ---
    p_home = sub.add_parser("go_home", help="Move all arm joints to zero")
    p_home.add_argument("--release", action="store_true", help="Release motor control after reaching home")
    p_home.add_argument("--velocity", type=float, default=5.0, help="Velocity limit (rad/s)")

    # --- release ---
    sub.add_parser("release", help="Release arm motor control (natural hang)")

    # --- set ---
    p_set = sub.add_parser("set", help="Send a custom 14-dim arm target")
    p_set.add_argument("joints", nargs=14, type=float,
                        metavar="q", help="14 joint angles in radians")
    p_set.add_argument("--velocity", type=float, default=5.0)
    p_set.add_argument("--release", action="store_true")

    # --- waypoints ---
    p_wp = sub.add_parser("waypoints", help="Move through waypoints from file")
    p_wp.add_argument("--file", required=True, help="Text file with one waypoint per line (14 floats)")
    p_wp.add_argument("--velocity", type=float, default=5.0)
    p_wp.add_argument("--tolerance", type=float, default=0.15)
    p_wp.add_argument("--release", action="store_true")

    # --- common ---
    parser.add_argument("--motion", action="store_true",
                        help="Use motion mode (rt/arm_sdk). Default is debug mode (rt/lowcmd), matching teleop script.")

    args = parser.parse_args()

    if args.command == "read":
        logger_mp.info("Read-only mode: subscribing to joint state (no commands sent) ...")
        read_joint_state_readonly()
        return

    motion_mode = args.motion
    logger_mp.info(f"Initializing G1_29_ArmController (motion_mode={motion_mode}) ...")
    ctrl = G1_29_ArmController(motion_mode=motion_mode)

    # Diagnostics
    mode_m = ctrl.get_mode_machine()
    logger_mp.info(f"Robot mode_machine = {mode_m}")
    logger_mp.info(f"Publishing to: {'rt/arm_sdk' if motion_mode else 'rt/lowcmd'}")
    logger_mp.info(f"arm_velocity_limit = {ctrl.arm_velocity_limit}")

    # Warm up: hold current arm position for 2s before sending new targets
    current_arm_q = ctrl.get_current_dual_arm_q()
    with ctrl.ctrl_lock:
        ctrl.q_target = current_arm_q.copy()
    logger_mp.info(f"Warming up (2s), holding current arm q = {np.round(current_arm_q, 3).tolist()}")
    time.sleep(2.0)

    # Verify arms held steady
    after_warmup_q = ctrl.get_current_dual_arm_q()
    delta = np.abs(after_warmup_q - current_arm_q)
    logger_mp.info(f"After warm-up, max joint drift = {delta.max():.4f} rad")
    if delta.max() > 0.05:
        logger_mp.warning("Arms drifted during warm-up — commands may not be reaching the robot!")

    need_hold = False

    try:
        if args.command == "go_home":
            # Natural hanging home: elbows at ~1.0 rad, rest at zero
            HOME_Q = [0, 0, 0, 1.0, 0, 0, 0,
                      0, 0, 0, 1.0, 0, 0, 0]
            print_all_joints(ctrl)
            logger_mp.info(f"Going home (velocity_limit={args.velocity}) ...")
            ctrl.ctrl_arm_through_waypoints(
                [HOME_Q],
                velocity_limit=args.velocity,
                tolerance=0.1,
            )
            logger_mp.info("Home reached.")
            print_all_joints(ctrl)
            need_hold = not args.release

        elif args.command == "release":
            print_all_joints(ctrl)
            ctrl.ctrl_dual_arm_release()
            logger_mp.info("Motor control released.")

        elif args.command == "set":
            target = np.array(args.joints)
            print_all_joints(ctrl)
            logger_mp.info(f"Setting arm target: {target}")
            logger_mp.info(f"Velocity limit: {args.velocity}")
            ctrl.arm_velocity_limit = args.velocity
            with ctrl.ctrl_lock:
                ctrl.q_target = target
            max_wait = 200
            for _ in range(max_wait):
                current = ctrl.get_current_dual_arm_q()
                if np.all(np.abs(current - target) < 0.1):
                    break
                time.sleep(0.05)
            logger_mp.info("Target reached (or timeout).")
            print_all_joints(ctrl)
            need_hold = not args.release

        elif args.command == "waypoints":
            wps = []
            with open(args.file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    vals = [float(x) for x in line.split()]
                    if len(vals) != 14:
                        logger_mp.error(f"Expected 14 values per line, got {len(vals)}: {line}")
                        return
                    wps.append(vals)
            logger_mp.info(f"Loaded {len(wps)} waypoints from {args.file}")
            print_all_joints(ctrl)
            ctrl.ctrl_arm_through_waypoints(wps, velocity_limit=args.velocity, tolerance=args.tolerance)
            print_all_joints(ctrl)
            need_hold = not args.release

        if need_hold:
            logger_mp.info("Holding position. Press Enter to release and exit, or Ctrl+C to exit immediately.")
            try:
                input()
            except EOFError:
                pass
            ctrl.ctrl_dual_arm_release()
            logger_mp.info("Released.")
        elif args.command != "release":
            if args.release:
                ctrl.ctrl_dual_arm_release()
                logger_mp.info("Released.")

    except KeyboardInterrupt:
        logger_mp.info("Interrupted, releasing...")
        ctrl.ctrl_dual_arm_release()
    finally:
        logger_mp.info("Done.")


if __name__ == "__main__":
    main()
