"""
Record or replay a safe arm+hand trajectory via teleoperation.

Usage:
    # Record
    python record_safe_trajectory.py record --output safe_traj.npz [--frequency 30] [--sim]

    # Replay on robot (forward then reverse)
    python record_safe_trajectory.py replay safe_traj.npz [--sim] [--speed 1.0]

    # Preview only (print stats + plot, no robot)
    python record_safe_trajectory.py preview safe_traj.npz

Controls (record mode):
    Left X (1st)  - start VR teleop (take control of robot)
    Right A (1st) - start recording trajectory
    Right A (2nd) - stop recording & save file (stays in teleop)
    Left X (2nd)  - release VR control (robot holds position)
    keyboard q    - exit program
"""

import argparse
import time
import threading
import numpy as np
from multiprocessing import Array, Lock

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import logging_mp
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

QUIT = False

def on_press(key):
    global QUIT
    if key == 'q':
        QUIT = True
        logger_mp.info("[keyboard] q: QUIT")


def print_trajectory_summary(traj_data, path=""):
    """Print trajectory stats for quick sanity check."""
    arm_q = traj_data["arm_q"]
    freq = float(traj_data["frequency"])
    n_frames = len(arm_q)
    duration = n_frames / freq

    print(f"\n{'='*60}")
    print(f"  Trajectory: {path}")
    print(f"  Frames: {n_frames}  |  Frequency: {freq:.0f} Hz  |  Duration: {duration:.2f} s")
    print(f"  Arm DOF: {arm_q.shape[1]}")
    print(f"  Start arm_q (L7+R7): {np.array2string(arm_q[0], precision=3, suppress_small=True)}")
    print(f"  End   arm_q (L7+R7): {np.array2string(arm_q[-1], precision=3, suppress_small=True)}")

    max_delta = np.max(np.abs(np.diff(arm_q, axis=0)), axis=0)
    print(f"  Max frame-to-frame delta per joint: {np.array2string(max_delta, precision=4)}")

    if "hand_state" in traj_data:
        hand = traj_data["hand_state"]
        print(f"  Hand DOF: {hand.shape[1]}")
        print(f"  Start hand: {np.array2string(hand[0], precision=3, suppress_small=True)}")
        print(f"  End   hand: {np.array2string(hand[-1], precision=3, suppress_small=True)}")
    print(f"{'='*60}\n")


def plot_trajectory(traj_data):
    """Plot arm joint angles over time."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot.")
        return

    arm_q = traj_data["arm_q"]
    freq = float(traj_data["frequency"])
    t = np.arange(len(arm_q)) / freq

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for j in range(7):
        axes[0].plot(t, arm_q[:, j], label=f"L_joint{j}")
    axes[0].set_ylabel("Left arm (rad)")
    axes[0].legend(fontsize=8, ncol=4)
    axes[0].grid(True, alpha=0.3)

    for j in range(7):
        axes[1].plot(t, arm_q[:, 7 + j], label=f"R_joint{j}")
    axes[1].set_ylabel("Right arm (rad)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(fontsize=8, ncol=4)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Safe Trajectory - Arm Joint Angles")
    plt.tight_layout()
    plt.show()


def do_preview(path):
    """Print stats and plot without connecting to robot."""
    traj = np.load(path)
    print_trajectory_summary(traj, path)
    plot_trajectory(traj)


def do_replay(path, sim, speed):
    """Load trajectory and replay on robot: forward, pause, then reverse."""
    from teleop.robot_control.robot_arm import G1_29_ArmController

    traj = np.load(path)
    print_trajectory_summary(traj, path)

    arm_q = traj["arm_q"]
    freq = float(traj["frequency"]) * speed
    zero_tau = np.zeros(len(arm_q[0]))

    arm_ctrl = G1_29_ArmController(motion_mode=False, simulation_mode=sim)

    HANG_HOME_Q = [0.5, 0, 0, 1.2, -1.5708, -0.7, 0,
                   0.5, 0, 0, 1.2, 1.5708, -0.7, 0]
    logger_mp.info("Moving arms to home position (natural hang) ...")
    arm_ctrl.ctrl_arm_through_waypoints([HANG_HOME_Q], velocity_limit=5.0, tolerance=0.1)
    logger_mp.info("Arms at home position.")

    def replay(indices, direction):
        logger_mp.info(f"Replaying {direction} ({len(arm_q)} frames at {freq:.0f} Hz) ...")
        for i in indices:
            t0 = time.perf_counter()
            arm_ctrl.ctrl_dual_arm(arm_q[i], zero_tau)
            time.sleep(max(0, 1.0 / freq - (time.perf_counter() - t0)))
        logger_mp.info(f"Replay {direction} done.")

    try:
        replay(range(len(arm_q)), "FORWARD")
        input("\nForward replay done. Press Enter to replay REVERSE (or Ctrl+C to abort)...")
        replay(range(len(arm_q) - 1, -1, -1), "REVERSE")
        logger_mp.info("Replay test complete. Robot should be back at start position.")
    except KeyboardInterrupt:
        logger_mp.info("Aborted.")
    finally:
        arm_ctrl.ctrl_dual_arm_release()


def do_record(output, frequency, sim):
    """Record trajectory via teleoperation (no camera/video, arm+hand only).

    Control flow:
        Left X   -> toggle VR teleop on/off
        Right A  -> 1st press: start recording; 2nd press: stop & save (stays in teleop)
        keyboard q -> exit program
    """
    global QUIT

    from televuer import TeleVuerWrapper
    from teleop.robot_control.robot_arm import G1_29_ArmController
    from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
    from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
    from sshkeyboard import listen_keyboard, stop_listening

    kb_thread = threading.Thread(
        target=listen_keyboard,
        kwargs={"on_press": on_press, "until": None, "sequential": False},
        daemon=True,
    )
    kb_thread.start()

    from multiprocessing import shared_memory
    dummy_img_shape = (480, 640, 3)
    dummy_shm = shared_memory.SharedMemory(create=True, size=np.prod(dummy_img_shape) * np.uint8().itemsize)
    tv_wrapper = TeleVuerWrapper(
        binocular=False, use_hand_tracking=False,
        img_shape=dummy_img_shape, img_shm_name=dummy_shm.name,
        return_state_data=True, return_hand_rot_data=False,
    )

    arm_ik = G1_29_ArmIK()
    arm_ctrl = G1_29_ArmController(motion_mode=False, simulation_mode=sim)

    HANG_HOME_Q = [0.5, 0, 0, 1.2, -1.5708, -0.7, 0,
                   0.5, 0, 0, 1.2, 1.5708, -0.7, 0]
    logger_mp.info("Moving arms to home position (natural hang) ...")
    arm_ctrl.ctrl_arm_through_waypoints([HANG_HOME_Q], velocity_limit=5.0, tolerance=0.1)
    logger_mp.info("Arms at home position.")

    arm_ctrl.ctrl_lower_body_to_zero(duration=2.0)

    left_hand_pos_array = Array('d', 75, lock=True)
    right_hand_pos_array = Array('d', 75, lock=True)
    left_dex3_cmd = Array('d', 7, lock=True)
    right_dex3_cmd = Array('d', 7, lock=True)
    dual_hand_data_lock = Lock()
    dual_hand_state_array = Array('d', 14, lock=False)
    dual_hand_action_array = Array('d', 14, lock=False)
    hand_ctrl = Dex3_1_Controller(
        left_hand_pos_array, right_hand_pos_array,
        dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array,
        simulation_mode=sim,
        left_cmd_q_in=left_dex3_cmd, right_cmd_q_in=right_dex3_cmd,
    )

    teleop_active = False
    recording = False
    saved = False
    arm_q_list = []
    hand_state_list = []
    prev_la = False
    prev_ra = False

    try:
        logger_mp.info("Waiting. Left X: start teleop | keyboard q: exit")
        while not QUIT:
            tele_data = tv_wrapper.get_motion_state_data()

            la = bool(getattr(tele_data.tele_state, 'left_aButton', False))
            ra = bool(getattr(tele_data.tele_state, 'right_aButton', False))

            # Left X rising edge -> toggle teleop (starts recording automatically)
            if la and not prev_la:
                teleop_active = not teleop_active
                if teleop_active:
                    arm_ctrl.speed_gradual_max()
                    arm_q_list.clear()
                    hand_state_list.clear()
                    recording = True
                    saved = False
                    logger_mp.info("[Left X] Teleop ON + RECORDING started")
                else:
                    logger_mp.info("[Left X] Teleop OFF - robot holds position. Press 'q' to exit.")

            # Right A rising edge -> stop recording & save (stay in teleop)
            if ra and not prev_ra and teleop_active and recording:
                recording = False
                if len(arm_q_list) > 0:
                    traj_data = {
                        "arm_q": np.array(arm_q_list),
                        "hand_state": np.array(hand_state_list),
                        "frequency": frequency,
                    }
                    np.savez(output, **traj_data)
                    saved = True
                    logger_mp.info(f"[Right A] RECORDING stopped & saved ({len(arm_q_list)} frames)")
                    print_trajectory_summary(traj_data, output)
                else:
                    logger_mp.info("[Right A] RECORDING stopped - no frames captured")

            prev_la, prev_ra = la, ra

            if not teleop_active:
                time.sleep(0.01)
                continue

            # Teleop loop
            t0 = time.time()
            current_q = arm_ctrl.get_current_dual_arm_q()
            current_dq = arm_ctrl.get_current_dual_arm_dq()
            sol_q, sol_tau = arm_ik.solve_ik(
                tele_data.left_arm_pose, tele_data.right_arm_pose,
                current_q, current_dq,
            )
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tau)

            if recording:
                with dual_hand_data_lock:
                    hand_state = np.array(dual_hand_state_array[:])
                arm_q_list.append(np.array(current_q))
                hand_state_list.append(hand_state)

            elapsed = time.time() - t0
            time.sleep(max(0, 1.0 / frequency - elapsed))

    except KeyboardInterrupt:
        logger_mp.info("Interrupted.")
    finally:
        stop_listening()
        arm_ctrl.ctrl_dual_arm_release()

        if not saved and len(arm_q_list) > 0:
            logger_mp.info("Unsaved recording discarded.")

        dummy_shm.close()
        dummy_shm.unlink()


def main():
    p = argparse.ArgumentParser(
        description="Record, replay, or preview a safe trajectory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    # record
    rec = sub.add_parser("record", help="Record a new trajectory via teleoperation")
    rec.add_argument('--output', type=str, required=True, help='Output .npz file')
    rec.add_argument('--frequency', type=float, default=30.0)
    rec.add_argument('--sim', action='store_true')

    # replay
    rep = sub.add_parser("replay", help="Replay trajectory on robot (forward then reverse)")
    rep.add_argument('file', type=str, help='.npz trajectory file')
    rep.add_argument('--sim', action='store_true')
    rep.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier')

    # preview
    pre = sub.add_parser("preview", help="Print stats and plot (no robot needed)")
    pre.add_argument('file', type=str, help='.npz trajectory file')

    args = p.parse_args()

    if args.command == "record":
        do_record(args.output, args.frequency, args.sim)
    elif args.command == "replay":
        do_replay(args.file, args.sim, args.speed)
    elif args.command == "preview":
        do_preview(args.file)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
