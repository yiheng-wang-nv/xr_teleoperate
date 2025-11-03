import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Array, Lock
import threading
import logging_mp
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
from teleop.image_server.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int,publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion  
STOP           = False  # Enable to begin system exit procedure
RECORD_TOGGLE  = False  # [Ready] ⇄ [Recording] ⟶ [AutoSave] ⟶ [Ready]         (⇄ manual) (⟶ auto)
RECORD_RUNNING = False  # True if [Recording]
RECORD_READY   = True   # True if [Ready], False if [Recording] / [AutoSave]
# task info
TASK_NAME = None
TASK_DESC = None
ITEM_ID = None
## Keyboard control for Dex3 thumbs (controller mode)
LEFT_DEX3_CMD_ARRAY = None  # will be set after arrays are created
RIGHT_DEX3_CMD_ARRAY = None # will be set after arrays are created
DEX3_LEFT_LIMITS = {
    "thumb0": (-1.04719755, 1.04719755),
    "thumb1": (-0.72431163, 1.04719755),
    "thumb2": (0.0, 1.74532925),
}
DEX3_KB_STEP = 0.05
# Left thumb1 target range
THUMB1_MIN_RAD = 0.0
THUMB1_MAX_RAD = 55.0 * np.pi / 180.0
# Right thumb1 target range
R_THUMB1_MIN_RAD = -55.0 * np.pi / 180.0
R_THUMB1_MAX_RAD = 0.0

# Controller per-press edge detector state
CONTROLLER_PREV = {
    "la": False,  # left A (Quest left controller shows as A/B in this wrapper)
    "lb": False,  # left B
    "ra": False,  # right A
    "rb": False,  # right B
    "lt": False,  # left trigger (index finger button)
    "rt": False,  # right trigger (index finger button)
}

def on_press(key):
    global STOP, START, RECORD_TOGGLE
    if key == 'r':
        START = not START
        logger_mp.info(f"[on_press] START -> {START}")
    elif key == 'q':
        STOP = True
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
    else:
        # Keyboard: left thumb1 c/v, right thumb1 b/n (only when START is True)
        if not START:
            logger_mp.debug("[on_press] START=False; ignore manual finger key")
            return
        if key in ('c','v') and LEFT_DEX3_CMD_ARRAY is not None:
            try:
                with LEFT_DEX3_CMD_ARRAY.get_lock():
                    cmd = np.array(LEFT_DEX3_CMD_ARRAY[:])
                    if key == 'c':
                        cmd[1] = float(THUMB1_MIN_RAD)
                    elif key == 'v':
                        cmd[1] = float(THUMB1_MAX_RAD)
                    LEFT_DEX3_CMD_ARRAY[:] = cmd
            except Exception as e:
                logger_mp.warning(f"[on_press] Failed to update left Dex3 via keyboard: {e}")
        elif key in ('b','n') and RIGHT_DEX3_CMD_ARRAY is not None:
            try:
                with RIGHT_DEX3_CMD_ARRAY.get_lock():
                    cmd = np.array(RIGHT_DEX3_CMD_ARRAY[:])
                    if key == 'b':
                        cmd[1] = float(R_THUMB1_MAX_RAD)
                    elif key == 'n':
                        cmd[1] = float(R_THUMB1_MIN_RAD)
                    RIGHT_DEX3_CMD_ARRAY[:] = cmd
            except Exception as e:
                logger_mp.warning(f"[on_press] Failed to update right Dex3 via keyboard: {e}")
        else:
            logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def on_release(key):
    pass

def on_info(info):
    """Only handle CMD_TOGGLE_RECORD's task info"""
    global TASK_NAME, TASK_DESC, ITEM_ID
    TASK_NAME   = info.get("task_name")
    TASK_DESC   = info.get("task_desc")
    ITEM_ID     = info.get("item_id")
    logger_mp.debug(f"[on_info] Updated globals: {TASK_NAME}, {TASK_DESC}, {ITEM_ID}")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, RECORD_READY
    return {
        "START": START,
        "STOP": STOP,
        "RECORD_RUNNING": RECORD_RUNNING,
        "RECORD_READY": RECORD_READY,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'save data\'s frequency')

    # basic control parameters
    parser.add_argument('--xr-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device tracking source')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pickup_tools', help = 'task name for recording')
    parser.add_argument('--task-desc', type = str, default = 'pick the tools from the left plate in order and place them in the right plate.', help = 'task goal for recording')

    args = parser.parse_args()
    logger_mp.info(f"args: {args}")

    try:
        # ipc communication. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press, on_info=on_info, get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, kwargs={"on_press": on_press, "on_release": on_release, "until": None, "sequential": False,}, daemon=True)
            listen_keyboard_thread.start()

        # image client: img_config should be the same as the configuration in image_server.py (of Robot's development computing unit)
        if args.sim:
            img_config = {
                'fps': 30,
                'head_camera_type': 'opencv',
                'head_camera_image_shape': [480, 640],  # Head camera resolution
                'head_camera_id_numbers': [0],
                'wrist_camera_type': 'opencv',
                'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
                'wrist_camera_id_numbers': [2, 4],
            }
        else:
            img_config = {
                'fps': 30,
                'head_camera_type': 'opencv',
                'head_camera_image_shape': [480, 640],  # Head camera resolution
                'head_camera_id_numbers': [0],
                'wrist_camera_type': 'opencv',
                'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
                'wrist_camera_id_numbers': [2, 4],
            }


        ASPECT_RATIO_THRESHOLD = 2.0 # If the aspect ratio exceeds this value, it is considered binocular
        if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
            BINOCULAR = True
        else:
            BINOCULAR = False
        if 'wrist_camera_type' in img_config:
            WRIST = True
        else:
            WRIST = False
        
        if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
            tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
        else:
            tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

        tv_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(tv_img_shape) * np.uint8().itemsize)
        tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = tv_img_shm.buf)

        if WRIST and args.sim:
            wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
            wrist_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(wrist_img_shape) * np.uint8().itemsize)
            wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = wrist_img_shm.buf)
            img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name, 
                                    wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name, server_address="127.0.0.1")
        elif WRIST and not args.sim:
            wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
            wrist_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(wrist_img_shape) * np.uint8().itemsize)
            wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = wrist_img_shm.buf)
            img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name, 
                                    wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name)
        else:
            img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name)

        image_receive_thread = threading.Thread(target = img_client.receive_process, daemon = True)
        image_receive_thread.daemon = True
        image_receive_thread.start()

        # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(binocular=BINOCULAR, use_hand_tracking=args.xr_mode == "hand", img_shape=tv_img_shape, img_shm_name=tv_img_shm.name, 
                                    return_state_data=True, return_hand_rot_data = False)

        # arm (fixed to G1-29)
        arm_ik = G1_29_ArmIK()
        arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)

        # end-effector (Dex3 only)
        left_hand_pos_array = Array('d', 75, lock = True)      # [input] hand tracking positions
        right_hand_pos_array = Array('d', 75, lock = True)     # [input]
        # Always create command arrays so keyboard/controller paths work
        left_dex3_cmd_q_array = Array('d', 7, lock = True)
        right_dex3_cmd_q_array = Array('d', 7, lock = True)
        with left_dex3_cmd_q_array.get_lock():
            # Order: [thumb0, thumb1, thumb2, middle0, middle1, index0, index1]
            left_init = np.array([
                0.0 * np.pi / 180.0,  # thumb0
                0.0,                    # thumb1 (controlled by c/v)
                0.0,                    # thumb2
                -60.0 * np.pi / 180.0,  # middle0
                -40.0 * np.pi / 180.0,  # middle1
                -60.0 * np.pi / 180.0,  # index0
                -40.0 * np.pi / 180.0,  # index1
            ], dtype=float)
            left_dex3_cmd_q_array[:] = left_init
        with right_dex3_cmd_q_array.get_lock():
            right_init = np.array([
                0.0 * np.pi / 180.0,  # thumb0
                0.0,                    # thumb1 (controlled by b/n)
                0.0,                    # thumb2
                60.0 * np.pi / 180.0,   # middle0 (placeholder)
                40.0 * np.pi / 180.0,   # middle1 (placeholder)
                60.0 * np.pi / 180.0,   # index0
                40.0 * np.pi / 180.0,   # index1
            ], dtype=float)
            right_dex3_cmd_q_array[:] = right_init
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left/right hand states
        dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left/right hand actions
        hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim,
                                      left_cmd_q_in=left_dex3_cmd_q_array, right_cmd_q_in=right_dex3_cmd_q_array)
        # expose arrays for keyboard/controller callbacks
        LEFT_DEX3_CMD_ARRAY = left_dex3_cmd_q_array
        RIGHT_DEX3_CMD_ARRAY = right_dex3_cmd_q_array
        
        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20) # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # controller + motion mode
        if args.xr_mode == "controller" and args.motion:
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
            sport_client = LocoClient()
            sport_client.SetTimeout(0.0001)
            sport_client.Init()
        
        # record + headless mode
        if args.record and args.headless:
            recorder = EpisodeWriter(task_dir = args.task_dir + args.task_name, task_goal = args.task_desc, frequency = args.frequency, rerun_log = False)
        elif args.record and not args.headless:
            recorder = EpisodeWriter(task_dir = args.task_dir + args.task_name, task_goal = args.task_desc, frequency = args.frequency, rerun_log = True)


        logger_mp.info("Press 'r' or Left A button to start/resume; 'r'/LA again to pause; 'q' to exit.")
        while not STOP:
            # Wait until START is True or STOP requested
            while not START and not STOP:
                # Allow controller Left A button to resume from pause
                if args.xr_mode == "controller":
                    try:
                        tele_data = tv_wrapper.get_motion_state_data()
                        la = bool(getattr(tele_data.tele_state, 'left_aButton', False))
                        if la and not CONTROLLER_PREV.get("la", False):
                            START = True
                            logger_mp.info("[controller] Left A: START -> True")
                        CONTROLLER_PREV["la"] = la
                    except Exception:
                        pass
                time.sleep(0.01)
            if STOP:
                break
            logger_mp.info("start program.")
            arm_ctrl.speed_gradual_max()
            while START and not STOP:
                start_time = time.time()

                if not args.headless:
                    tv_resized_image = cv2.resize(tv_img_array, (tv_img_shape[1] // 2, tv_img_shape[0] // 2))
                    cv2.imshow("record image", tv_resized_image)
                    # opencv GUI communication
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        STOP = True
                        if args.sim:
                            publish_reset_category(2, reset_pose_publisher)
                    elif key == ord('s'):
                        RECORD_TOGGLE = True
                    elif key == ord('a'):
                        if args.sim:
                            publish_reset_category(2, reset_pose_publisher)
                    # Note: 'r' is handled by the keyboard/IPC handler to avoid double toggles here

                if args.record and RECORD_TOGGLE:
                    RECORD_TOGGLE = False
                    if not RECORD_RUNNING:
                        if recorder.create_episode():
                            RECORD_RUNNING = True
                        else:
                            logger_mp.error("Failed to create episode. Recording not started.")
                    else:
                        RECORD_RUNNING = False
                        recorder.save_episode()
                        if args.sim:
                            publish_reset_category(1, reset_pose_publisher)
                # get input data
                tele_data = tv_wrapper.get_motion_state_data()
                # Controller bindings - use A buttons for START/RECORD
                if args.xr_mode == "controller":
                    la = bool(getattr(tele_data.tele_state, 'left_aButton', False))
                    ra = bool(getattr(tele_data.tele_state, 'right_aButton', False))
                    # Left A button rising edge -> toggle START
                    if la and not CONTROLLER_PREV.get("la", False):
                        prev = START
                        START = not START
                        if prev != START:
                            logger_mp.info(f"[controller] Left A: START -> {START}")
                    # Right A button rising edge -> toggle RECORD when running
                    if START and ra and not CONTROLLER_PREV.get("ra", False):
                        RECORD_TOGGLE = True
                        logger_mp.info("[controller] Right A: RECORD_TOGGLE -> True")
                    # Update previous states
                    CONTROLLER_PREV["la"], CONTROLLER_PREV["ra"] = la, ra
                if args.xr_mode == "hand":
                    with left_hand_pos_array.get_lock():
                        left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                    with right_hand_pos_array.get_lock():
                        right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
                elif args.xr_mode == "controller":
                    # Dex3 controller mode - Trigger continuous control
                    # Read current command arrays
                    with left_dex3_cmd_q_array.get_lock():
                        left_cmd = np.array(left_dex3_cmd_q_array[:])
                    with right_dex3_cmd_q_array.get_lock():
                        right_cmd = np.array(right_dex3_cmd_q_array[:])

                    # Get trigger values
                    # Note: televuer returns trigger_value in range [10.0, 0.0] (inverted)
                    # We need to normalize to [0.0, 1.0]
                    left_trigger_raw = getattr(tele_data, 'left_trigger_value', 10.0)
                    right_trigger_raw = getattr(tele_data, 'right_trigger_value', 10.0)
                    
                    # Normalize: 10.0 -> 0.0 (not pressed), 0.0 -> 1.0 (fully pressed)
                    left_trigger = np.clip((10.0 - left_trigger_raw) / 10.0, 0.0, 1.0)
                    right_trigger = np.clip((10.0 - right_trigger_raw) / 10.0, 0.0, 1.0)
                    
                    # Map trigger values to thumb1 range
                    # Left trigger: 0.0 -> THUMB1_MIN_RAD (closed), 1.0 -> THUMB1_MAX_RAD (open)
                    left_cmd[1] = THUMB1_MIN_RAD + left_trigger * (THUMB1_MAX_RAD - THUMB1_MIN_RAD)
                    
                    # Right trigger: 0.0 -> R_THUMB1_MAX_RAD (closed), 1.0 -> R_THUMB1_MIN_RAD (open)
                    # (note: right hand has inverted range)
                    right_cmd[1] = R_THUMB1_MAX_RAD + right_trigger * (R_THUMB1_MIN_RAD - R_THUMB1_MAX_RAD)

                    # Final safety clipping for all joints
                    left_cmd[0] = np.clip(left_cmd[0], *DEX3_LEFT_LIMITS["thumb0"])
                    left_cmd[1] = np.clip(left_cmd[1], *DEX3_LEFT_LIMITS["thumb1"])
                    left_cmd[2] = np.clip(left_cmd[2], *DEX3_LEFT_LIMITS["thumb2"])

                    with left_dex3_cmd_q_array.get_lock():
                        left_dex3_cmd_q_array[:] = left_cmd
                    with right_dex3_cmd_q_array.get_lock():
                        right_dex3_cmd_q_array[:] = right_cmd
        
            
                # high level control
                if args.xr_mode == "controller" and args.motion:
                    # quit teleoperate
                    if tele_data.tele_state.right_aButton:
                        STOP = True
                    # command robot to enter damping mode. soft emergency stop function
                    if tele_data.tele_state.left_thumbstick_state and tele_data.tele_state.right_thumbstick_state:
                        sport_client.Damp()
                    # control, limit velocity to within 0.3
                    sport_client.Move(-tele_data.tele_state.left_thumbstick_value[1]  * 0.3,
                                      -tele_data.tele_state.left_thumbstick_value[0]  * 0.3,
                                      -tele_data.tele_state.right_thumbstick_value[0] * 0.3)

                # get current robot state data.
                current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
                current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

                # solve ik using motor data and wrist pose, then use ik results to control arms.
                time_ik_start = time.time()
                sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_arm_pose, tele_data.right_arm_pose, current_lr_arm_q, current_lr_arm_dq)
                time_ik_end = time.time()
                logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

                # record data
                if args.record:
                    RECORD_READY = recorder.is_ready()
                # dex hand state/action logging
                if args.xr_mode in ("hand", "controller"):
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []
                # head image
                current_tv_image = tv_img_array.copy()
                # wrist image
                if WRIST:
                    current_wrist_image = wrist_img_array.copy()
                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if BINOCULAR:
                        colors[f"color_{0}"] = current_tv_image[:, :tv_img_shape[1]//2]
                        colors[f"color_{1}"] = current_tv_image[:, tv_img_shape[1]//2:]
                        if WRIST:
                            colors[f"color_{2}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                            colors[f"color_{3}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                    else:
                        colors[f"color_{0}"] = current_tv_image
                        if WRIST:
                            colors[f"color_{1}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                            colors[f"color_{2}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / args.frequency) - time_elapsed)
                time.sleep(sleep_time)
                logger_mp.debug(f"main process sleep: {sleep_time}")

            if not STOP:
                logger_mp.info("Paused. Waiting for 'r' to resume or 'q' to exit.")
                # Stop residual body motion in controller+motion mode
                if args.xr_mode == "controller" and args.motion:
                    try:
                        sport_client.Move(0.0, 0.0, 0.0)
                    except Exception:
                        pass

    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting program...")
    finally:
        arm_ctrl.ctrl_dual_arm_go_home()

        if args.ipc:
            ipc_server.stop()
        else:
            stop_listening()
            listen_keyboard_thread.join()

        if args.sim:
            sim_state_subscriber.stop_subscribe()
        tv_img_shm.close()
        tv_img_shm.unlink()
        if WRIST:
            wrist_img_shm.close()
            wrist_img_shm.unlink()

        if args.record:
            recorder.close()
        logger_mp.info("Finally, exiting program.")
        exit(0)
