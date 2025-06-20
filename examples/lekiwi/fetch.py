from lerobot.common.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.common.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.common.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.utils.control_utils import predict_action
from lerobot.common.utils.utils import get_safe_torch_device
import time


robot_config = LeKiwiClientConfig(remote_ip="192.168.1.12", id="lekiwi_101")

print("Loading the policy...")
policy = SmolVLAPolicy.from_pretrained("vladfatu/lekiwi_101")
# policy = SmolVLAPolicy.from_pretrained("70mmy/lekiwi_ball_10")
policy.reset()

print("Connecting to the robot...")
robot = LeKiwiClient(robot_config)
robot.connect()

print("Loading the dataset features...")
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

print("Sending search for the ball...")
robot.send_search_for_object('sports ball')

time.sleep(20)  # Wait for the robot to find the ball

print("Running inference")
start_time = time.time()
while time.time() - start_time < 15:
    obs = robot.get_observation()

    observation_frame = build_dataset_frame(dataset_features, obs, prefix="observation")
    # print(f"Observation frame: {observation_frame}")
    action_values = predict_action(
        observation_frame, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
    )
    action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
    robot.send_action(action)

print("Taking the ball back...")
robot.send_return_to_sender()

time.sleep(25)


robot.disconnect()

