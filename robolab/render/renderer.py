import math
import pygame

from Box2D import (
    b2Vec2,
    b2World,
)

from Box2D.b2 import (
    staticBody,
    dynamicBody,
    kinematicBody,
    polygonShape,
    circleShape,
    edgeShape,
)


class Renderer:

    _COLOR_BACKGROUND = (0, 0, 0, 0)

    _COLORS = {
        staticBody: (220, 220, 220, 255),
        dynamicBody: (127, 127, 127, 255),
        kinematicBody: (127, 127, 230, 255),
    }

    color_assets = (0, 255, 0, 255)

    flip_x_axis = False
    flip_y_axis = True

    def __init__(
        self,
        screen: pygame.Surface,
        scale: int,
    ) -> None:
        self.screen = screen
        self.scale = scale
        screen_width, screen_height = self.screen.get_size()
        self.screen_size = b2Vec2(screen_width, screen_height)
        self.screen_offset = b2Vec2(-0.01 * screen_width, -0.01 * screen_height)
        self._install()

    def _install(self):
        """Installs drawing methods for world objects."""
        edgeShape.draw = self._draw_edge
        polygonShape.draw = self._draw_polygon
        circleShape.draw = self._draw_circle

    def _transform_vertices(self, vertices: tuple):
        """Transforms points of vertices to pixel coordinates."""
        return [self._to_screen(vertex) for vertex in vertices]

    def _to_screen(self, point: b2Vec2) -> tuple[int, int]:
        """Transforms point from simulation to screen coordinates."""
        pos_x = point.x * self.scale - self.screen_offset.x
        pos_y = point.y * self.scale - self.screen_offset.y

        if self.flip_x_axis:
            pos_x = self.screen_size.x - pos_x
        if self.flip_y_axis:
            pos_y = self.screen_size.y - pos_y

        return int(pos_x), int(pos_y)

    def _draw_circle(self, body, fixture, color=None, width: int = 1) -> None:
        center = self._to_screen(body.position)
        radius = self.scale * fixture.shape.radius
        pygame.draw.circle(
            surface=self.screen,
            color=color or self.color_assets,
            center=center,
            radius=radius,
            width=width,
        )
        # Add roll indicator line to circle.
        radius = fixture.shape.radius
        x = math.cos(body.angle) * radius
        y = math.sin(body.angle) * radius
        border = self._to_screen(body.position + b2Vec2(x, y))
        pygame.draw.line(
            surface=self.screen,
            color=color or self.color_assets,
            start_pos=center,
            end_pos=border,
            width=width,
        )

    def _draw_polygon(self, body, fixture):
        polygon = fixture.shape
        transform = body.transform
        vertices = [transform * vertex for vertex in polygon.vertices]
        vertices = self._transform_vertices(vertices)
        # Draw edge.
        pygame.draw.polygon(self.screen, self._COLORS[body.type], vertices, 1)
        # Draw face.
        ## edge_color = [0.5 * color for color in self._COLORS[body.type]]
        ## pygame.draw.polygon(self.screen, edge_color, vertices, 0)  # Face.

    def _draw_edge(self, body, fixture):
        edge = fixture.shape
        vertices = [body.transform * edge.vertex1, body.transform * edge.vertex2]
        vertex1, vertex2 = self._transform_vertices(vertices)
        pygame.draw.line(self.screen, self._COLORS[body.type], vertex1, vertex2)

    def render(self, world: b2World) -> None:
        self.screen.fill(self._COLOR_BACKGROUND)
        for body in world.bodies:
            for fixture in body.fixtures:
                fixture.shape.draw(body, fixture)
