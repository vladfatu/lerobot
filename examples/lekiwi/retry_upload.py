import time

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import hw_to_dataset_features
from lerobot.common.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.common.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.common.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.common.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
import time

NB_CYCLES_CLIENT_CONNECTION = 750

leader_arm_config = SO101LeaderConfig(port="/dev/tty.usbmodem5A460849101", id="leader_101")
leader_arm = SO101Leader(leader_arm_config)

keyboard_config = KeyboardTeleopConfig()
keyboard = KeyboardTeleop(keyboard_config)

robot_config = LeKiwiClientConfig(remote_ip="192.168.8.191", id="lekiwi_101")
robot = LeKiwiClient(robot_config)

action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

dataset = LeRobotDataset(
    repo_id="vladfatu/lekiwi_ball1749918394",
    robot_type=robot.name,
    force_cache_sync=True
)

print("Uploading dataset to the hub")

dataset.push_to_hub()
