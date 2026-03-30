# 1. install conda
wget --inet4-only https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 2. install git lfs
sudo apt update
sudo apt install git-lfs

# 3. install unitree xr teleoperate
https://github.com/unitreerobotics/xr_teleoperate?tab=readme-ov-file#1--installation

# 4. note: when install isaaclab, git clone should use http command in a new machine
git clone https://github.com/isaac-sim/IsaacLab.git

git clone https://github.com/unitreerobotics/unitree_sim_isaaclab.git

# 5. note: export cyclonedds
export CYCLONEDDS_HOME="/home/nvidia/workspace/yiheng/cyclonedds/install"

# 6. install libstdcxx-ng

conda install -c conda-forge libstdcxx-ng

# connect unitree
sudo ip addr add 192.168.123.222/24 dev eno1 # only if reboot the machine
ssh unitree@192.168.123.164
# if not find dds
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eno1"/></Interfaces></General></Domain></CycloneDDS>'
# then use follow command to check
cyclonedds ps
python image_server/image_server.py

# image client view 
conda activate tv
cd /home/nvidia/workspace/yiheng/xr_teleoperate/teleop
python image_server/image_client.py


# start teleoperate on simulation
conda activate unitree_sim_env
cd /home/nvidia/workspace/yiheng/unitree_sim_isaaclab
python sim_main.py --device cuda --enable_cameras --task Isaac-Simple-Wave-G129-Dex3-Joint --enable_dex3_dds --robot_type g129


python sim_main.py --device cuda --enable_cameras --task Isaac-PickPlace-Surgical-G129-Dex3-Joint --enable_dex3_dds --robot_type g129
# headless 
python sim_main.py --device cuda --enable_cameras --task Isaac-PickPlace-Surgical-G129-Dex3-Joint --enable_dex3_dds --robot_type g129 --headless
# for headless, need to use image client
conda activate tv
cd /home/nvidia/workspace/yiheng/xr_teleoperate/teleop
python image_server/image_client.py --server-address="127.0.0.1" --hide-wrist --display-scale 2 --image-show
python image_server/image_client.py --server-address="127.0.0.1" --display-scale 1.3 --image-show


# start teleoperate on real
conda activate tv
cd /home/nvidia/workspace/yiheng/xr_teleoperate/teleop
python teleop_dex3_controller.py --record --task-name debug --task-desc "install trocar from tray"
# sim
python teleop_dex3_controller.py --record --task-name debug_rl --task-desc "install trocar from tray" --sim --thumb-mode keyboard
# sim controller
python teleop_dex3_controller.py --record --task-name Isaac-PickPlace-Surgical-G129-Dex3-Joint --task-desc "install trocar from tray" --sim --thumb-mode controller

# real with motion
L2+B -> L2 + UP -> R1 + x
python teleop_hand_and_arm.py \
  --xr-mode controller --arm G1_29 --ee dex3 --record --motion --task-name debug_motion

# real robot, controller, pick cotton ball
python teleop_hand_and_arm.py \
  --xr-mode controller --arm G1_29 --ee dex3 --record \
  --task-name get_cotton_fixed \
  --right-idx-middle-0-angle 30.0 \
  --right-idx-middle-1-angle 40.0 \
  --disable-clamp-right \
  --enable-clamp-left \
  --low-bandwidth

# trocar
python teleop_hand_and_arm.py \
  --xr-mode controller --arm G1_29 --ee dex3 --record \
  --task-name install_trocar_v3 \
  --right-idx-middle-0-angle 60.0 \
  --right-idx-middle-1-angle 40.0 \
  --left-thumb0-angle 0.0 \
  --right-thumb0-angle -27.5 \
  --enable-clamp-left \
  --enable-clamp-right \
  --low-bandwidth
  
# tube
python teleop_hand_and_arm.py \
  --xr-mode controller --arm G1_29 --ee dex3 --record \
  --task-name pick_tube \
  --right-idx-middle-0-angle 60.0 \
  --right-idx-middle-1-angle 40.0 \
  --right-thumb1-min-angle -55.0 \
  --enable-clamp-left \
  --enable-clamp-right \
  --low-bandwidth

# without xr device, use
python teleop_hand_and_arm.py \
  --xr-mode controller --arm G1_29 --ee dex3 --sim --headless --record

  
# real robot
sudo ip addr add 192.168.123.222/24 dev eno2
ssh unitree@192.168.123.164  
python image_server/image_server.py

# process recorded data and convert to lerobot format
conda activate unitree_lerobot
cd ~/workspace/yiheng/unitree_IL_lerobot/
bash process_recorded_data.sh <dir of the recorded data> <task-name>
# example:
# bash process_recorded_data.sh /home/nvidia/workspace/yiheng/xr_teleoperate/teleop/utils/data/debug_rl debug_rl_lerobot


# replay
python unitree_lerobot/eval_robot/repaly_robot.py \
    --repo_id="i4h/interve_data" \
    --root="" \
    --episodes=0 \
    --frequency=30 \
    --arm="G1_29" \
    --ee="dex3"


https://docs.google.com/spreadsheets/d/1ND93zwxU1DG9yZzTLXD1mIGJ2FcwYd-_zGDS-acan3c/edit?gid=0#gid=0
