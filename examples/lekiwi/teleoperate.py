from lerobot.common.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.common.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.common.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

robot_config = LeKiwiClientConfig(remote_ip="192.168.8.191", id="lekiwi_101")

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
teleop_arm.connect()
telep_keyboard.connect()

while True:
    observation = robot.get_observation()

    arm_action = teleop_arm.get_action()
    arm_action = {f"arm_{k}": v for k, v in arm_action.items()}

    keyboard_keys = telep_keyboard.get_action()
    base_action = robot._from_keyboard_to_base_action(keyboard_keys)

    robot.send_action(arm_action | base_action)
