import time

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import hw_to_dataset_features
from lerobot.common.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.common.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.common.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.common.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
import time

NB_EPISODES = 2
EPISODE_LENGTH = 15  # seconds
RESET_TIME = 7  # seconds

# Use this action
# DEFAULT_ARM_ACTION = {'arm_shoulder_pan.pos': -5.0, 'arm_shoulder_lift.pos': -98.92428630533719, 'arm_elbow_flex.pos': 99.27895448400182, 'arm_wrist_flex.pos': 19.973137973137966, 'arm_wrist_roll.pos': -0.31746031746031633, 'arm_gripper.pos': 30.567244829886591}
INACTIVE_BASE_ACTION = {'x.vel': 0.0, 'y.vel': 0.0, 'theta.vel': 0.0}

leader_arm_config = SO101LeaderConfig(port="/dev/tty.usbmodem5A460849101", id="leader_101")
leader_arm = SO101Leader(leader_arm_config)

keyboard_config = KeyboardTeleopConfig()
keyboard = KeyboardTeleop(keyboard_config)

robot_config = LeKiwiClientConfig(remote_ip="192.168.1.13", id="lekiwi_101")
robot = LeKiwiClient(robot_config)

action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

dataset = LeRobotDataset.create(
    repo_id="vladfatu/lekiwi_ball" + str(int(time.time())),
    fps=10,
    features=dataset_features,
    robot_type=robot.name,
)

leader_arm.connect()
keyboard.connect()
robot.connect()

if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
    exit()

print("Saving initial arm position to reset after each episode")
initial_observation = robot.get_observation()
initial_arm_action = {'arm_shoulder_pan.pos': initial_observation['arm_shoulder_pan.pos'], 'arm_shoulder_lift.pos': initial_observation['arm_shoulder_lift.pos'], 'arm_elbow_flex.pos': initial_observation['arm_elbow_flex.pos'], 'arm_wrist_flex.pos': initial_observation['arm_wrist_flex.pos'], 'arm_wrist_roll.pos': initial_observation['arm_wrist_roll.pos'], 'arm_gripper.pos': initial_observation['arm_gripper.pos']}

print("Starting LeKiwi recording")

# This loop will run for NB_EPISODES episodes
for i in range(NB_EPISODES):
    print(f"Starting episode {i + 1}")
    # This loop will run for EPISODE_LENGTH seconds
    start_time = time.time()
    while time.time() - start_time < EPISODE_LENGTH:
        arm_action = leader_arm.get_action()
        arm_action = {f"arm_{k}": v for k, v in arm_action.items()}

        keyboard_keys = keyboard.get_action()

        base_action = robot._from_keyboard_to_base_action(keyboard_keys)

        action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action

        action_sent = robot.send_action(action)
        observation = robot.get_observation()

        frame = {**action_sent, **observation}
        task = "Grab the ball and hold it"

        dataset.add_frame(frame, task)
        i += 1

    print("Saving dataset episode")
    dataset.save_episode()


    # reset the robot to a default state and sleep for RESET_TIME seconds before the next cycle
    robot.send_action(initial_arm_action | INACTIVE_BASE_ACTION)
    time.sleep(RESET_TIME)

print("Disconnecting Teleop Devices and LeKiwi Client")
robot.disconnect()
leader_arm.disconnect()
keyboard.disconnect()

print("Uploading dataset to the hub")

dataset.push_to_hub()
