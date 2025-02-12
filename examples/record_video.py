from gymnasium.wrappers import RecordVideo
from robolab.envs.robot_arm import RobotArm
from robolab.envs.robot_lab import RobotLab


def record_video_arm() -> None:
    env = RobotArm(render_mode="rgb_array", task_id=0, max_episode_steps=400)
    env = RecordVideo(env, video_folder="videos", name_prefix="arm")
    observation, info = env.reset()
    is_running = True
    while is_running:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            is_running = False
    env.close()


def record_video_lab() -> None:
    env = RobotLab(render_mode="rgb_array", task_id=0, x_range=6, y_range=4, max_episode_steps=400)
    env = RecordVideo(env, video_folder="videos", name_prefix="lab")
    observation, info = env.reset()
    is_running = True
    while is_running:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if any(terminated.values()) or any(truncated.values()):
            is_running = False
    env.close()


if __name__ == "__main__":
    record_video_arm()
    record_video_lab()
