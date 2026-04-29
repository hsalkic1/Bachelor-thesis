import json
import os
import random
import numpy as np
from trajectory import T_REPLAN

MAPS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_maps.json")
FIXED_OBSTACLE_COUNT = True
V_MAX = 10
MAX_STEPS = 150


def _trajectory_covers_point(x, y, z, w, h, d, vx, vy, vz, px, py, pz, safe_r, map_size):
    ox, oy, oz = float(x), float(y), float(z)
    for _ in range(MAX_STEPS + 1):
        dx = max(ox - px, 0, px - (ox + w))
        dy = max(oy - py, 0, py - (oy + h))
        dz = max(oz - pz, 0, pz - (oz + d))
        if np.sqrt(dx*dx + dy*dy + dz*dz) < safe_r:
            return True
        ox = float(np.clip(ox + vx * T_REPLAN, 0, map_size[0] - w))
        oy = float(np.clip(oy + vy * T_REPLAN, 0, map_size[1] - h))
        oz = float(np.clip(oz + vz * T_REPLAN, 0, map_size[2] - d))
    return False


def generate_and_save_random_maps(filename, n_maps, map_size):
    margin_x = int(0.08 * map_size[0])
    margin_y = int(0.08 * map_size[1])
    margin_z = int(0.08 * map_size[2])
    w = margin_x
    h = margin_y
    d = margin_z
    safe_radius = 0.05 * map_size[0]
    min_dist = np.sqrt(map_size[0]**2 + map_size[1]**2 + map_size[2]**2) / 2
    map_data = []

    for i in range(10, n_maps + 10):
        start_x = start_y = start_z = 0
        goal_x = goal_y = goal_z = 0
        while np.sqrt((start_x - goal_x)**2 + (start_y - goal_y)**2 + (start_z - goal_z)**2) < min_dist:
            start_x = random.randint(margin_x, map_size[0] - margin_x)
            start_y = random.randint(margin_y, map_size[1] - margin_y)
            start_z = random.randint(margin_z, map_size[2] - margin_z)
            goal_x = random.randint(margin_x, map_size[0] - margin_x)
            goal_y = random.randint(margin_y, map_size[1] - margin_y)
            goal_z = random.randint(margin_z, map_size[2] - margin_z)

        obstacles = []
        if FIXED_OBSTACLE_COUNT:
            i = n_maps

        while len(obstacles) < i:
            x = random.randint(0, map_size[0] - w)
            y = random.randint(0, map_size[1] - h)
            z = random.randint(0, map_size[2] - d)
            v_x = random.randint(-V_MAX, V_MAX)
            v_y = random.randint(-V_MAX, V_MAX)
            v_z = random.randint(-V_MAX, V_MAX)

            if (_trajectory_covers_point(x, y, z, w, h, d, v_x, v_y, v_z, start_x, start_y, start_z, safe_radius, map_size)
                    or _trajectory_covers_point(x, y, z, w, h, d, v_x, v_y, v_z, goal_x, goal_y, goal_z, safe_radius, map_size)):
                continue
            obstacles.append((x, y, z, w, h, d, v_x, v_y, v_z))

        map_data.append({
            "start": (start_x, start_y, start_z),
            "goal": (goal_x, goal_y, goal_z),
            "obstacles": obstacles,
            "map_size": map_size,
        })

    with open(filename, 'w') as f:
        json.dump(map_data, f, indent=4)
    print(f"Map saved to {filename}")


if __name__ == "__main__":
    generate_and_save_random_maps(MAPS_FILE, 30, (200, 200, 200))
