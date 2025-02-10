from typing import Any, Optional

import gymnasium as gym
import numpy
import pygame

from Box2D import (
    b2Vec2,
    b2World,
)

from robolab.envs.robotic_arm import RoboticArm
from robolab.render.renderer import Renderer

FPS: int = 60
TIME_STEP: float = 1.0 / FPS
VELOCITY_ITERATIONS: int = 10  # Iterations to compute next velocity.
POSITION_ITERATIONS: int = 10  # Iterations to compute next position.
SCALE: int = 8


class RobotLab(gym.Env):
    """
    This class runs robotic arms in parallel using a single PyBox2D instance. The Box2D world
    gets populated with a grid of size M x N with robotic arms. This allows arms to be processed in
    parallel and accelerates data collection.

    # Action Space

    The action space consists of the action space of each robotic arm.

    # Observation Space

    The observation space consists of the observation space of each robotic arm.

    # Episode Termination

    The episode (rollout) is terminated either if one of the robotic arms completes its task
    successfully or if the number of episode steps is exceeded (episode gets truncated).

    # Credits
    Created by Kai Fischer (2025)
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        world: Optional[b2World] = None,
        x_range: int = 5,
        y_range: int = 4,
        xy_gap: float = 1.0,
        x_min: float = 0.0,
        x_max: float = 60.0,
        y_min: float = 0.0,
        y_max: float = 40.0,
        gravity: b2Vec2 = b2Vec2(0.0, -9.8),
        max_motor_speed: float = 10.0,
        max_motor_torque: float = 10.0,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
        task_id: Optional[int] = 1,
    ) -> None:
        self.isopen = True

        if world is None:
            self.world = b2World(gravity=gravity)
            self.is_single = True
        else:
            self.world = world
            self.is_single = False

        x_diam = x_max - x_min
        y_diam = y_max - y_min

        self.x_diam = x_range * x_diam
        self.y_diam = y_range * y_diam

        coords, ids = self._get_coords_of_boxes(x_range, y_range, x_diam, y_diam, xy_gap)

        self.robots = []
        for uid, (x, y) in zip(ids, coords):
            self.robots.append(
                RoboticArm(
                    uid=uid,
                    world=self.world,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                    x_center=x,
                    y_center=y,
                    max_motor_speed=max_motor_speed,
                    max_motor_torque=max_motor_torque,
                    max_episode_steps=max_episode_steps,
                    task_id=task_id,
                    render_mode=None,  # (!)
                )
            )

        self.action_space = gym.spaces.Dict(
            {robot.uid: robot.action_space for robot in self.robots}
        )

        self.observation_space = gym.spaces.Dict(
            {robot.uid: robot.observation_space for robot in self.robots}
        )

        self.render_mode = render_mode
        self.clock = None
        self.screen: Optional[pygame.Surface] = None

    @staticmethod
    def _get_coords_of_boxes(
        x_range: int,
        y_range: int,
        x_diam: float,
        y_diam: float,
        dist_gap: float,
    ) -> tuple[list[tuple[float, float]], list[str]]:
        uids = []
        coords = []
        for j in range(y_range):
            for i in range(x_range):
                x_pos = i * (x_diam + dist_gap)
                y_pos = j * (y_diam + dist_gap)
                coords.append((x_pos, y_pos))
                uids.append(f"{i}_{j}")
        return coords, uids

    def step(
        self,
        actions: dict[str, numpy.array],
    ) -> tuple[
        dict[str, numpy.array], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]
    ]:
        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}

        for robot, action in zip(self.robots, actions.values()):
            observation, reward, terminated, truncated, info = robot.step(action=action)
            observations[robot.uid] = observation
            rewards[robot.uid] = reward
            terminateds[robot.uid] = terminated
            truncateds[robot.uid] = truncated
            infos[robot.uid] = info

        self.world.Step(TIME_STEP, VELOCITY_ITERATIONS, POSITION_ITERATIONS)
        self.world.ClearForces()

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminateds, truncateds, infos

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, numpy.array], dict[str, Any]]:
        super().reset(seed=seed)
        observations = {}
        infos = {}
        for robot in self.robots:
            obs, info = robot.reset(seed=seed, options=options)
            observations[robot.uid] = obs
            infos[robot.uid] = info
        if self.render_mode == "human":
            self.render()
        return observations, infos

    def render(self) -> None:
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("RoboLab")
            self.screen = pygame.display.set_mode(
                (
                    int(1.02 * SCALE * self.x_diam),
                    int(1.02 * SCALE * self.y_diam),
                )
            )
            self.clock = pygame.time.Clock()
            self.renderer = Renderer(screen=self.screen, scale=SCALE)

        self.renderer.render(world=self.world)
        self.clock.tick(FPS)
        pygame.event.pump()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                exit()

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def debug():
    from Box2D.examples.framework import Framework, main

    class RobotLabDebug(Framework):

        def __init__(self):
            super().__init__()
            self.robot_lab = RobotLab(world=self.world)
            self.setCenter(value=(0.25 * self.robot_lab.x_diam, 0.25 * self.robot_lab.y_diam))
            self.setZoom(zoom=5)

        def Step(self, settings) -> None:
            super().Step(settings)
            actions = self.robot_lab.action_space.sample()
            observations, rewards, terminateds, truncateds, infos = self.robot_lab.step(
                actions=actions
            )
            if any(terminateds.values()) or any(truncateds.values()):
                observation, infos = self.robot_lab.reset()

    main(RobotLabDebug)


if __name__ == "__main__":
    debug()
