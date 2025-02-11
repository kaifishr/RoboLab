import gymnasium as gym
from robolab.envs.robot_arm import RobotArm


def test_env(env: gym.Env, max_num_rollouts: int = 2) -> None:
    total_reward = 0.0
    is_running = True
    rollout = 0
    while is_running:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            observation, info = env.reset()
            rollout += 1
            if rollout == max_num_rollouts:
                break
    print(f"{total_reward = }")
    env.close()


if __name__ == "__main__":
    env = RobotArm(render_mode="human")
    test_env(env=env)
