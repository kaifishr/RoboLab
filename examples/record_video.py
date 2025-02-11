from gymnasium.wrappers import RecordVideo
from robolab.envs.robot_arm import RobotArm


def record_video() -> None:
    env = RobotArm(render_mode="rgb_array", task_id=0)
    env = RecordVideo(env, video_folder="videos")  # , episode_trigger=lambda x: True)
    observation, info = env.reset()
    is_running = True
    while is_running:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            is_running = False
    env.close()


if __name__ == "__main__":
    record_video()
