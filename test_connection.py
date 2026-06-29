from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower

config = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="my_awesome_follower_arm",
)

follower = SO101Follower(config)
follower.connect(calibrate=False)
follower.calibrate()
follower.disconnect()