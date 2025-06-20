from lerobot.common.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.common.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.common.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

SEARCH_ARM_ACTION = {'arm_shoulder_pan.pos': -5.0, 'arm_shoulder_lift.pos': -98.92428630533719, 'arm_elbow_flex.pos': 99.27895448400182, 'arm_wrist_flex.pos': 19.973137973137966, 'arm_wrist_roll.pos': -0.31746031746031633, 'arm_gripper.pos': 0.867244829886591}

robot_config = LeKiwiClientConfig(remote_ip="192.168.1.12", id="lekiwi_101")

teleop__arm_config = SO101LeaderConfig(
    port="/dev/tty.usbmodem5A460849101",
    id="leader_101",
)

teleop_keyboard_config = KeyboardTeleopConfig(
    id="mac_keyboard",
)

robot = LeKiwiClient(robot_config)
teleop_arm = SO101Leader(teleop__arm_config)
telep_keyboard = KeyboardTeleop(teleop_keyboard_config)
robot.connect()
telep_keyboard.connect()

try:
    teleop_arm.connect()
except ConnectionError as e:
    print(f"Failed to connect teleop arm: Running without teleop arm. Error: {e}")

while True:
    observation = robot.get_observation()

    if teleop_arm.is_connected:
        arm_action = teleop_arm.get_action()
        arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
    else:
        arm_action = SEARCH_ARM_ACTION

    keyboard_keys = telep_keyboard.get_action()
    
    if robot.teleop_keys["search"] in keyboard_keys:
        robot.send_search_for_object('fork')
    else:
        base_action = robot._from_keyboard_to_base_action(keyboard_keys)

        robot.send_action(arm_action | base_action)
