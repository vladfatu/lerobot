from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.common.utils.control_utils import predict_action
from lerobot.common.utils.utils import get_safe_torch_device
import time

NB_CYCLES_CLIENT_CONNECTION = 1000

robot_config = LeKiwiClientConfig(remote_ip="192.168.10.19", id="lekiwi_101")
robot = LeKiwiClient(robot_config)

robot.connect()

# policy = ACTPolicy.from_pretrained("70mmy/lekiwi_ball_4")
policy = SmolVLAPolicy.from_pretrained("vladfatu/lekiwi_101")
policy.reset()


action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}
# obs_features = hw_to_dataset_features(robot.observation_features, "observation")

print("Running inference")
start_time = time.time()
while time.time() - start_time < 30:
    obs = robot.get_observation()

    # print(f"Observation features: {dataset_features}")
    # print("=============================")
    # print(f"Observation: {obs}")
    # print("=============================")

    observation_frame = build_dataset_frame(dataset_features, obs, prefix="observation")
    # print(f"Observation frame: {observation_frame}")
    action_values = predict_action(
        observation_frame, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
    )
    action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
    robot.send_action(action)

robot.disconnect()
