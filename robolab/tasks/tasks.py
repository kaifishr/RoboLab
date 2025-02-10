import random

from typing import Optional

from Box2D import (
    b2Body,
    b2CircleShape,
    b2EdgeShape,
    b2FixtureDef,
    b2PolygonShape,
    b2Vec2,
    b2World,
)


class Task:

    def __init__(
        self,
        world: b2World,
        origin: b2Vec2,
        bodies: list[b2Body],
        bodies_position: list[b2Vec2],
        noise: Optional[float] = 1.0,
    ):
        self.world = world
        self.origin = origin
        self.bodies = bodies
        self.bodies_position = bodies_position
        self.noise = noise

    @staticmethod
    def _reset(
        body: b2Body,
        position: b2Vec2,
        angle: float = 0.0,
        noise: Optional[float] = None,
    ) -> None:
        if noise is not None:
            position += b2Vec2(random.uniform(a=-noise, b=noise), 0.0)
        body.transform = (position, angle)
        body.linearVelocity = b2Vec2(0.0, 0.0)
        body.angularVelocity = 0.0

    def reset(self) -> None:
        for body, position in zip(self.bodies, self.bodies_position):
            self._reset(body=body, position=position)


class StackBoxes(Task):
    """Creates two boxes to stack."""

    def __init__(
        self,
        world: b2World,
        origin: b2Vec2,
        x_diam: float,
        y_diam: float,
        bodies: list[b2Body],
        bodies_position: list[b2Vec2],
        noise: Optional[float] = None,
        box_size: float = 2.0,
        box_density: float = 1.0,
        box_friction: float = 1.0,
        eps_pos: float = 1e-1,
        eps_vel: float = 1e-1,
    ) -> None:
        super().__init__(
            world=world,
            origin=origin,
            bodies=bodies,
            bodies_position=bodies_position,
            noise=noise,
        )

        self.size = box_size
        self.eps_pos = eps_pos
        self.eps_vel = eps_vel
        density = box_density
        friction = box_friction

        shape_1 = b2PolygonShape(box=(self.size, self.size))
        fixture = b2FixtureDef(
            shape=shape_1,
            density=density,
            friction=friction,
        )

        pos_1 = b2Vec2(0.25 * x_diam, self.size)
        position = self.origin + pos_1
        self.box_1 = self.world.CreateDynamicBody(position=position, fixtures=fixture)
        bodies.append(self.box_1)
        bodies_position.append(position)

        shape_2 = b2PolygonShape(box=(self.size, self.size))
        fixture = b2FixtureDef(
            shape=shape_2,
            density=density,
            friction=friction,
        )

        pos_2 = b2Vec2(0.75 * x_diam, self.size)
        position = self.origin + pos_2
        self.box_2 = self.world.CreateDynamicBody(position=position, fixtures=fixture)
        bodies.append(self.box_2)
        bodies_position.append(position)

    def _is_vertical_aligned(self, pos_1: b2Vec2, pos_2: b2Vec2) -> bool:
        return (pos_2.x - self.size) < pos_1.x < (pos_2.x + self.size)

    @staticmethod
    def _is_box_1_on_box_2(pos_1: b2Vec2, pos_2: b2Vec2, size: float, eps: float) -> bool:
        return (pos_2.y + size - eps) < (pos_1.y - size) < (pos_2.y + size + eps)

    @staticmethod
    def _is_box_2_on_box_1(pos_1: b2Vec2, pos_2: b2Vec2, size: float, eps: float) -> bool:
        return (pos_1.y + size - eps) < (pos_2.y - size) < (pos_1.y + size + eps)

    def _is_stacked(self, pos_1: b2Vec2, pos_2: b2Vec2, size: float) -> bool:
        if self._is_box_1_on_box_2(
            pos_1=pos_1,
            pos_2=pos_2,
            size=size,
            eps=self.eps_pos,
        ) or self._is_box_2_on_box_1(
            pos_1=pos_1,
            pos_2=pos_2,
            size=size,
            eps=self.eps_pos,
        ):
            return True
        return False

    def _is_at_rest(self) -> bool:
        return (
            self.box_1.linearVelocity.length < self.eps_vel
            and self.box_2.linearVelocity.length < self.eps_vel
        )

    def _reward_stack_boxes(self) -> tuple[float, bool]:
        reward = 0.0
        terminated = False

        box_1_pos = self.box_1.position - self.origin
        box_2_pos = self.box_2.position - self.origin

        if self._is_vertical_aligned(pos_1=box_1_pos, pos_2=box_2_pos):
            reward += 0.1
            if self._is_stacked(pos_1=box_1_pos, pos_2=box_2_pos, size=self.size):
                reward += 10.0
                if self._is_at_rest():
                    reward += 100.0
                    terminated = True

        return reward, terminated

    def _reward_distance_boxes(self, weight: float = 1.0) -> float:
        return -weight * ((self.box_1.position - self.box_2.position).length - 2.0 * self.size)

    def comp_reward(self) -> tuple[float, bool]:
        reward, terminated = self._reward_stack_boxes()
        reward += self._reward_distance_boxes()
        return reward, terminated


class DropBallIntoBox(Task):

    def __init__(
        self,
        world: b2World,
        origin: b2Vec2,
        x_diam: float,
        y_diam: float,
        bodies: list[b2Body],
        bodies_position: list[b2Vec2],
        noise: Optional[float] = None,
        ball_radius: float = 1.5,
        ball_density: float = 0.5,
        ball_friction: float = 2.0,
        box_width: float = 8.0,
        box_height: float = 8.0,
    ):
        super().__init__(
            world=world,
            origin=origin,
            bodies=bodies,
            bodies_position=bodies_position,
            noise=noise,
        )

        self.box_width = box_width
        self.box_height = box_height

        ball_offset = b2Vec2(0.25 * x_diam, ball_radius)
        self.ball_position = self.origin + ball_offset

        box_offset = b2Vec2(0.75 * x_diam, 0.0)
        self.box_position = self.origin + box_offset

        # Creates a dynamic ball.
        fixture = b2FixtureDef(
            shape=b2CircleShape(radius=ball_radius),
            density=ball_density,
            friction=ball_friction,
        )
        # position = self.origin + self.ball_position
        self.body = self.world.CreateDynamicBody(position=self.ball_position, fixtures=fixture)
        bodies.append(self.body)
        bodies_position.append(self.ball_position)

        # Create the static box.
        self.box_min_x = -0.5 * self.box_width
        self.box_max_x = 0.5 * self.box_width
        box_shapes = [
            b2EdgeShape(vertices=[(self.box_min_x, 0.0), (self.box_min_x, self.box_height)]),
            b2EdgeShape(vertices=[(self.box_max_x, 0.0), (self.box_max_x, self.box_height)]),
        ]

        self.box = self.world.CreateStaticBody(
            position=self.box_position,
            shapes=box_shapes,
        )

        self.box_opening = self.box_position + b2Vec2(0.0, self.box_height)

    def _reward_ball_is_in_box(self) -> tuple[float, bool]:
        ball_position = self.body.position - self.origin
        box_position = self.box_position - self.origin
        x_min = box_position.x - 0.5 * self.box_width
        x_max = box_position.x + 0.5 * self.box_width
        reward = 0.0
        terminated = False
        if x_min < ball_position.x < x_max and ball_position.y < self.box_height:
            reward = 100.0
            terminated = True
        return reward, terminated

    def _reward_ball_distance_to_opening(self, weight: float = 1e-3) -> float:
        return -weight * (self.box_opening - self.body.position).length

    def comp_reward(self) -> tuple[float, bool]:
        reward, terminated = self._reward_ball_is_in_box()
        reward += self._reward_ball_distance_to_opening()
        return reward, terminated
