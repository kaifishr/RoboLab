"""Robotic arm environment for reinforcement learning."""

import math
from typing import Any, Optional

import gymnasium as gym
import numpy

import pygame

from Box2D import (
    b2EdgeShape,
    b2FixtureDef,
    b2PolygonShape,
    b2Vec2,
    b2Body,
    b2World,
)
from Box2D.examples.framework import (
    Framework,
    main,
)

from roboarm.tasks.tasks import (
    StackBoxes,
    DropBallIntoBox,
)

from roboarm.render.renderer import Renderer


FPS: int = 60
TIME_STEP: float = 1.0 / FPS
VELOCITY_ITERATIONS: int = 20  # Iterations to compute next velocity.
POSITION_ITERATIONS: int = 20  # Iterations to compute next position.
SCALE: int = 16


class RoboticArm(gym.Env):
    """
    This is a simple 5-joint robotic arm.

    There are two tasks for the robotic arm:
    - Stacking boxes.
    - Putting a ball into a bowl.

    # Action Space

    Actions are continuous motor speed values in the range [-1, 1] for each of the five joints.

    # Observation Space

    The observation space consists of the state of the robotic arm as well as the state of the task.

    The robotic arm's state consists of 39 floating point values. 
    There are five joints and three links.

    Joints:
    - angle in radians
    - motor speed
    - speed of joint

    Links:
    - angle in radians
    - angular speed
    - linear speed x
    - linear speed y
    - position x
    - position y

    The observation space depends on the task.

    For each box in task 1:
    - angle in radians
    - angular speed
    - linear speed x
    - linear speed y
    - position x
    - position y

    For the ball in task 2:
    - angle in radians
    - angular speed
    - linear speed x
    - linear speed y
    - position x
    - position y

    # Rewards

    There are two tasks with different rewards. Task 1 consists of stacking boxes. Task 2 consists
    of putting a ball into a box. The reward increases if the boxes are getting closer or if the
    ball is getting closer to the box. There is a reward of 100 for stacking one box on top of the
    other or putting the ball into the box.

    # Starting State

    The robotic arm starts always with the same position. There is a noise parameter to vary the
    boxes' position as well as the ball's position.

    # Episode Termination

    The episode (rollout) is terminated either if the task is completed successfully or if the
    number of episode steps is exceeded (episode gets truncated).

    # Credits
    Created by Kai Fischer (2025)
    """

    _link_vertices = [
        (1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
        (1.0, -1.0),
    ]

    _base_vertices = [
        (1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
        (1.0, -1.0),
    ]

    metadata = {
        "render_modes": ["human"],
        "render_fps": FPS,
    }

    _MAX_SPEED_JOINT = 10.0

    def __init__(
        self,
        world: Optional[b2World] = None,
        uid: str = "0",
        x_center: float = 0.0,
        y_center: float = 0.0,
        x_min: float = 0.0,
        x_max: float = 60.0,
        y_min: float = 0.0,
        y_max: float = 40.0,
        density: float = 1.0,
        friction: float = 1.0,
        gravity: b2Vec2 = (0.0, -9.8),
        max_motor_speed: float = 10.0,
        max_motor_torque: float = 10.0,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
        task_id: int = 1,
    ) -> None:
        self.isopen = True

        self.world = world
        if self.world is None:
            self.world = b2World(gravity=gravity)

        self.uid = uid
        self.center_x = x_center
        self.center_y = y_center
        self.origin = b2Vec2(x_center, y_center)
        self.x_diam = x_max - x_min
        self.y_diam = y_max - y_min
        self.x_norm = 1.0 / self.x_diam
        self.y_norm = 1.0 / self.y_diam

        self.max_motor_speed = max_motor_speed
        self.max_motor_torque = max_motor_torque
        self.max_episode_steps = max_episode_steps

        base_size_x = 2.0  # NOTE: This is half block diameter!
        base_size_y = 1.0

        link_half_width = 0.6
        link_half_length = 8.0

        link_1_size_x = link_half_length
        link_1_size_y = link_half_width
        link_2_size_x = link_half_width
        link_2_size_y = 0.8 * link_half_length
        link_3_size_x = link_half_width
        link_3_size_y = 0.7 * link_half_length

        self._make_robot_case(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            x_center=x_center,
            y_center=y_center,
        )

        self.joints = []
        self.fixtures = []
        self.links = []
        self.positions = []  # Initial link positions.

        # TODO: Move this to Task class.
        self.task_objects = []
        self.task_obj_pos = []  # Initial / reset object positions of task.

        # TODO
        # self._make_robot_arm()
        # self._make_robot_gripper()

        # Robot base (static object)
        position = b2Vec2(x_center + 0.5 * self.x_diam, y_center + self.y_diam - base_size_y)
        vertices = [(base_size_x * x, base_size_y * y) for x, y in self._base_vertices]
        body_shape = b2PolygonShape(vertices=vertices)
        body_fixture_def = b2FixtureDef(shape=body_shape)
        robot_base = self.world.CreateStaticBody(position=position)
        robot_base.CreateFixture(body_fixture_def)

        # Link 1 (dynamic object)
        position = robot_base.position + b2Vec2(-link_1_size_x, -base_size_y)
        self.link_1 = self.world.CreateDynamicBody(position=position)
        vertices = [(link_1_size_x * x, link_1_size_y * y) for x, y in self._link_vertices]
        link_shape = b2PolygonShape(vertices=vertices)
        link_fixture_def = b2FixtureDef(shape=link_shape, density=density, friction=friction)
        self.link_1.CreateFixture(link_fixture_def)
        self.links.append(self.link_1)
        self.fixtures.append(link_fixture_def)
        self.positions.append(position)

        # Link 2 (dynamic body)
        position = self.link_1.position + b2Vec2(-link_1_size_x, -link_2_size_y)
        self.link_2 = self.world.CreateDynamicBody(position=position)
        vertices = [(link_2_size_x * x, link_2_size_y * y) for x, y in self._link_vertices]
        link_shape = b2PolygonShape(vertices=vertices)
        link_fixture_def = b2FixtureDef(shape=link_shape, density=density, friction=friction)
        self.link_2.CreateFixture(link_fixture_def)
        self.links.append(self.link_2)
        self.fixtures.append(link_fixture_def)
        self.positions.append(position)

        # Link 3 (dynamic body)
        position = self.link_2.position + b2Vec2(0.0, -link_2_size_y - link_3_size_y)
        self.link_3 = self.world.CreateDynamicBody(position=position)
        vertices = [(link_3_size_x * x, link_3_size_y * y) for x, y in self._link_vertices]
        link_shape = b2PolygonShape(vertices=vertices)
        link_fixture_def = b2FixtureDef(shape=link_shape, density=density, friction=friction)
        self.link_3.CreateFixture(link_fixture_def)
        self.links.append(self.link_3)
        self.fixtures.append(link_fixture_def)
        self.positions.append(position)

        # Create gripper (dynamic body)
        position = self.link_3.position + b2Vec2(0.0, -link_3_size_y)
        self.left_finger = self._make_finger(
            position=position,
            is_left=True,
        )
        self.links.append(self.left_finger)
        self.positions.append(position)
        self.fixtures.append(self.left_finger.fixtures)

        position = self.link_3.position + b2Vec2(0.0, -link_3_size_y)
        self.right_finger = self._make_finger(
            position=position,
            is_left=False,
        )
        self.positions.append(position)
        self.links.append(self.right_finger)
        self.fixtures.append(self.right_finger.fixtures)

        # Weld base to link 1
        self.joint_1 = self.world.CreateRevoluteJoint(
            bodyA=robot_base,
            bodyB=self.link_1,
            anchor=robot_base.position + b2Vec2(0.0, -base_size_y),
            lowerAngle=-math.pi,  # -90 degrees
            upperAngle=math.pi,  # 90 degrees
            enableLimit=True,  # Enables limits set in `lowerAngle` and `upperAngle`.
            maxMotorTorque=max_motor_torque,
            motorSpeed=0.0,
            enableMotor=True,
            collideConnected=False,
        )
        self.joints.append(self.joint_1)

        # Weld link 1 to link 2
        self.joint_2 = self.world.CreateRevoluteJoint(
            bodyA=self.link_1,
            bodyB=self.link_2,
            anchor=self.link_1.position + b2Vec2(-link_1_size_x, 0.0),
            lowerAngle=-2.0 * math.pi,  # -90 degrees
            upperAngle=math.pi,  # 90 degrees
            enableLimit=True,  # Enables limits set in `lowerAngle` and `upperAngle`.
            maxMotorTorque=max_motor_torque,
            motorSpeed=0.0,
            enableMotor=True,
            collideConnected=False,
        )
        self.joints.append(self.joint_2)

        # Connect link 2 and link 3
        self.joint_3 = self.world.CreateRevoluteJoint(
            bodyA=self.link_2,
            bodyB=self.link_3,
            anchor=self.link_2.position + b2Vec2(0.0, -link_2_size_y),
            lowerAngle=-math.pi,
            upperAngle=math.pi,
            enableLimit=True,  # Enables limits set in `lowerAngle` and `upperAngle`.
            maxMotorTorque=max_motor_torque,
            motorSpeed=0.0,
            enableMotor=True,
            collideConnected=False,
        )
        self.joints.append(self.joint_3)

        # Connect link 3 and left finger
        self.joint_left_finger = self.world.CreateRevoluteJoint(
            bodyA=self.link_3,
            bodyB=self.left_finger,
            anchor=self.link_3.position + b2Vec2(0.0, -link_3_size_y),
            lowerAngle=-0.2 * math.pi,  # maximum left extension
            upperAngle=0.1 * math.pi,  # maxiumum right extension
            enableLimit=True,  # Enables limits set in `lowerAngle` and `upperAngle`.
            maxMotorTorque=max_motor_torque,
            motorSpeed=0.0,
            enableMotor=True,
            collideConnected=False,
        )
        self.joints.append(self.joint_left_finger)

        # Connect link 3 and right finger
        self.joint_right_finger = self.world.CreateRevoluteJoint(
            bodyA=self.link_3,
            bodyB=self.right_finger,
            anchor=self.link_3.position + b2Vec2(0.0, -link_3_size_y),
            lowerAngle=-0.1 * math.pi,  # maximum left extension
            upperAngle=0.2 * math.pi,  # maxiumum right extension
            enableLimit=True,  # Enables limits set in `lowerAngle` and `upperAngle`.
            maxMotorTorque=max_motor_torque,
            motorSpeed=0.0,
            enableMotor=True,
            collideConnected=False,
        )
        self.joints.append(self.joint_right_finger)

        if task_id == 0:
            self.task = StackBoxes(
                world=self.world,
                origin=self.origin,
                x_diam=self.x_diam,
                y_diam=self.y_diam,
                bodies=self.task_objects,
                bodies_position=self.task_obj_pos,
            )
        else:
            self.task = DropBallIntoBox(
                world=self.world,
                origin=self.origin,
                x_diam=self.x_diam,
                y_diam=self.y_diam,
                bodies=self.task_objects,
                bodies_position=self.task_obj_pos,
            )

        self.observation_space = self._make_observation_space()
        self.action_space = self._make_action_space()
        self.episode_step = 0

        self.render_mode = render_mode
        self.clock = None
        self.screen: Optional[pygame.Surface] = None

    def _make_observation_space(self) -> gym.spaces.Box:

        obs_joint = [
            -1.0,  # Angle in radians.
            -self.max_motor_speed,  # Radians per second.
            -100.0,  # Speed of joint.
        ]

        obs_links = [
            -1.0,  # Angle in radians.
            -100.0,  # Angular speed.
            -100.0,  # Linear speed x.
            -100.0,  # Linear speed y.
            -1.0,  # Position x.
            -1.0,  # Position y.
        ]

        obs_task = [
            -1.0,  # Angle in radians.
            -100.0,  # Angular speed.
            -100.0,  # Linear speed x.
            -100.0,  # Linear speed y.
            -1.0,  # Normalized position x.
            -1.0,  # Normalized position y.
        ]

        low = numpy.array(
            len(self.joints) * obs_joint
            + len(self.links) * obs_links
            + len(self.task_objects) * obs_task
        ).astype(numpy.float32)

        high = -1.0 * low

        observation_space = gym.spaces.Box(low=low, high=high)

        return observation_space

    def _make_action_space(self) -> gym.spaces.Box:
        low = len(self.joints) * [-1.0]
        high = len(self.joints) * [1.0]
        low = numpy.array(low, dtype=numpy.float32)
        high = numpy.array(high, dtype=numpy.float32)
        action_space = gym.spaces.Box(low=low, high=high)
        return action_space

    def _make_robot_case(self, x_min, x_max, y_min, y_max, x_center, y_center) -> None:
        shapes = [
            b2EdgeShape(vertices=[(x_min, y_min), (x_min, y_max)]),
            b2EdgeShape(vertices=[(x_min, y_max), (x_max, y_max)]),
            b2EdgeShape(vertices=[(x_max, y_max), (x_max, y_min)]),
            b2EdgeShape(vertices=[(x_max, y_min), (x_min, y_min)]),
        ]

        self.world.CreateStaticBody(
            position=(x_center, y_center),
            shapes=shapes,
        )

    def _make_finger(self, position: b2Vec2, is_left: bool = True) -> b2Body:

        alpha = 1.0 if is_left else -1.0

        link_1_vertices = [
            (alpha * -3.0, -3.0),
            (alpha * -2.0, -3.0),
            (alpha * 0.0, 0.0),
            (alpha * -1.0, 0.0),
        ]

        link_2_vertices = [
            (alpha * -3.0, -3.0),
            (alpha * -2.0, -3.0),
            (alpha * -1.0, -5.0),
        ]

        finger = self.world.CreateDynamicBody(position=position)
        link_1_shape = b2PolygonShape(vertices=link_1_vertices)
        link_2_shape = b2PolygonShape(vertices=link_2_vertices)
        _ = finger.CreateFixture(shape=link_1_shape, density=0.1, friction=2.0)
        _ = finger.CreateFixture(shape=link_2_shape, density=0.1, friction=2.0)
        return finger

    def _rotate(self, vec: b2Vec2, theta: float) -> b2Vec2:
        x = vec.x * math.cos(theta) - vec.y * math.sin(theta)
        y = vec.x * math.sin(theta) + vec.y * math.cos(theta)
        return b2Vec2(x, y)

    @staticmethod
    def _reset_body(
        body: b2Body,
        position: b2Vec2,
        angle: float = 0.0,
    ) -> None:
        body.transform = (position, angle)
        body.linearVelocity = b2Vec2(0.0, 0.0)
        body.angularVelocity = 0.0

    def _comp_reward(self) -> tuple[float, bool]:
        reward, terminated = self.task.comp_reward()
        return reward, terminated

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.sin(angle % (2.0 * math.pi))

    def _get_observation(self) -> numpy.array:

        obs = []

        for joint in self.joints:
            obs.append(self._normalize_angle(angle=joint.angle))
            obs.append(joint.motorSpeed)
            obs.append(joint.speed)

        for link in self.links:
            obs.append(self._normalize_angle(angle=link.angle))
            obs.append(link.angularVelocity)
            obs.append(link.linearVelocity.x)
            obs.append(link.linearVelocity.y)
            position = link.position - self.origin
            obs.append(position.x * self.x_norm)
            obs.append(position.y * self.y_norm)

        for box in self.task_objects:
            obs.append(self._normalize_angle(angle=box.angle))
            obs.append(box.angularVelocity)
            obs.append(box.linearVelocity.x)
            obs.append(box.linearVelocity.y)
            position = box.position - self.origin
            obs.append(position.x * self.x_norm)
            obs.append(position.y * self.y_norm)

        obs = numpy.array(obs).astype(numpy.float32)
        return obs

    def _apply_action(self, joint_actions: numpy.array) -> None:
        # TODO: Here we could also distinguish between link joints and gripper joints.
        # TODO: self.joints -> self.joints_links, self.joints_gripper
        for joint, act in zip(self.joints, joint_actions):

            # Easy: Set only target angular velocity of joint.
            joint.motorSpeed = float(self._MAX_SPEED_JOINT * act)

            # Alternative:
            # Hard: Set torque (rotational force) to achieve target angular velocity.
            # joint.motorSpeed = float(self._MAX_SPEED_JOINT * numpy.sign(act))
            # joint.maxMotorTorque = float(self.max_motor_torque * 0.5 * (1.0 + act))

    def step(self, action: numpy.array) -> tuple[numpy.array, float, bool, bool, dict[str, Any]]:
        self.episode_step += 1
        self._apply_action(joint_actions=action)
        self.world.Step(TIME_STEP, VELOCITY_ITERATIONS, POSITION_ITERATIONS)
        self.world.ClearForces()  # TODO: Check this.
        observation = self._get_observation()
        reward, terminated = self._comp_reward()
        truncated = True if self.episode_step > self.max_episode_steps else False
        info = {"uid": self.uid}

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[numpy.array, dict[str, Any]]:
        super().reset(seed=seed)
        self.episode_step = 0

        # Reset robotic arm.
        # self.arm.reset()
        for joint in self.joints:
            joint.motorSpeed = 0.0

        for link, position in zip(self.links, self.positions):
            self._reset_body(body=link, position=position)

        self.task.reset()

        observation = self._get_observation()
        info = {"uid": self.uid}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def render(self) -> None:
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                pygame.display.init()
                pygame.display.set_caption("Robot Arm")
                self.screen = pygame.display.set_mode(
                    (
                        int(1.05 * SCALE * self.x_diam),
                        int(1.05 * SCALE * self.y_diam),
                    )
                )
                self.clock = pygame.time.Clock()
                self.renderer = Renderer(screen=self.screen, scale=SCALE)

            self.renderer.render(world=self.world)
            self.clock.tick(FPS)
            pygame.event.pump()
            pygame.display.flip()

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class RoboticArmDebug(Framework):

    def __init__(self):
        super().__init__()
        self.robot = RoboticArm(world=self.world)
        self.viewCenter = (0.5 * self.robot.x_diam, 0.5 * self.robot.y_diam)

    def Step(self, settings) -> None:
        super().Step(settings)
        action = self.robot.action_space.sample()
        observation, reward, terminated, truncated, info = self.robot.step(action=action)
        if terminated or truncated:
            observation, infos = self.robot.reset()


if __name__ == "__main__":
    main(RoboticArmDebug)
