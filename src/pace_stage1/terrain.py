"""Frozen LeggedGym mixed-terrain construction adapted without changing its values."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TerrainConfig


@dataclass
class TerrainMap:
    height_field_raw: np.ndarray
    env_origins: np.ndarray
    vertices: np.ndarray
    triangles: np.ndarray
    horizontal_scale: float
    vertical_scale: float
    border_pixels: int
    env_length: float
    env_width: float


def build_terrain(config: TerrainConfig) -> TerrainMap:
    """Build the exact five-way terrain family from LeggedGym 8fa29acc.

    Isaac Gym is imported lazily so pure Stage 1 spec/tests do not depend on it.
    """
    from isaacgym import terrain_utils

    proportions = np.cumsum(config.terrain_proportions)
    width_pixels = int(config.terrain_width_m / config.horizontal_scale_m)
    length_pixels = int(config.terrain_length_m / config.horizontal_scale_m)
    border = int(config.border_size_m / config.horizontal_scale_m)
    total_rows = config.num_rows * length_pixels + 2 * border
    total_cols = config.num_cols * width_pixels + 2 * border
    height_field = np.zeros((total_rows, total_cols), dtype=np.int16)
    origins = np.zeros((config.num_rows, config.num_cols, 3), dtype=np.float32)

    for col in range(config.num_cols):
        for row in range(config.num_rows):
            difficulty = row / config.num_rows
            choice = col / config.num_cols + 0.001
            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=width_pixels,
                length=length_pixels,
                vertical_scale=config.vertical_scale_m,
                horizontal_scale=config.horizontal_scale_m,
            )
            slope = difficulty * 0.4
            step_height = 0.05 + 0.18 * difficulty
            obstacle_height = 0.05 + difficulty * 0.2
            if choice < proportions[0]:
                if choice < proportions[0] / 2.0:
                    slope *= -1.0
                terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.0)
            elif choice < proportions[1]:
                terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.0)
                terrain_utils.random_uniform_terrain(
                    terrain,
                    min_height=-0.05,
                    max_height=0.05,
                    step=0.005,
                    downsampled_scale=0.2,
                )
            elif choice < proportions[3]:
                if choice < proportions[2]:
                    step_height *= -1.0
                terrain_utils.pyramid_stairs_terrain(
                    terrain,
                    step_width=0.31,
                    step_height=step_height,
                    platform_size=3.0,
                )
            else:
                terrain_utils.discrete_obstacles_terrain(
                    terrain,
                    max_height=obstacle_height,
                    min_size=1.0,
                    max_size=2.0,
                    num_rects=20,
                    platform_size=3.0,
                )

            start_x = border + row * length_pixels
            end_x = start_x + length_pixels
            start_y = border + col * width_pixels
            end_y = start_y + width_pixels
            height_field[start_x:end_x, start_y:end_y] = terrain.height_field_raw
            x1 = int((config.terrain_length_m / 2.0 - 1.0) / config.horizontal_scale_m)
            x2 = int((config.terrain_length_m / 2.0 + 1.0) / config.horizontal_scale_m)
            y1 = int((config.terrain_width_m / 2.0 - 1.0) / config.horizontal_scale_m)
            y2 = int((config.terrain_width_m / 2.0 + 1.0) / config.horizontal_scale_m)
            origins[row, col] = (
                (row + 0.5) * config.terrain_length_m,
                (col + 0.5) * config.terrain_width_m,
                np.max(terrain.height_field_raw[x1:x2, y1:y2]) * config.vertical_scale_m,
            )

    vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
        height_field,
        config.horizontal_scale_m,
        config.vertical_scale_m,
        config.slope_threshold,
    )
    return TerrainMap(
        height_field_raw=height_field,
        env_origins=origins,
        vertices=vertices,
        triangles=triangles,
        horizontal_scale=config.horizontal_scale_m,
        vertical_scale=config.vertical_scale_m,
        border_pixels=border,
        env_length=config.terrain_length_m,
        env_width=config.terrain_width_m,
    )
