from moving_obstacles_runner import main, CLEARANCE_THRESHOLD
import numpy as np
import json
import os

MAPS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_maps.json")

with open(MAPS_FILE, 'r') as f:
    all_maps=json.load(f)

map_total = []
map_rrt = []
map_plane = []
map_q = []

all_pf_t_rrt = []
all_pf_t_plane = []
all_pf_t_plane_opt = []
all_pf_t_q = []
all_pf_t_total = []
all_pf_did_rrt = []
all_pf_did_plane_opt = []

clrs = []
reached_all = []

for i, map_data in enumerate(all_maps):
    obstacles = map_data["obstacles"]
    map_size = map_data["map_size"]
    start = map_data["start"]
    goal = map_data["goal"]

    print(f"\nMAP_{i+1}:")
    result = main(i+1, obstacles, start, goal, map_size)
    print(f"Map {i+1}: reached={result['reached_goal']}, steps={result['steps']}")

    map_total.append(sum(result['pf_t_total']))
    map_rrt.append(sum(result['pf_t_rrt']))
    map_plane.append(sum(result['pf_t_plane']))
    map_q.append(sum(result['pf_t_q']))

    all_pf_t_rrt.extend(result['pf_t_rrt'])
    all_pf_t_plane.extend(result['pf_t_plane'])
    all_pf_t_plane_opt.extend(result['pf_t_plane_opt'])
    all_pf_t_q.extend(result['pf_t_q'])
    all_pf_t_total.extend(result['pf_t_total'])
    all_pf_did_rrt.extend(result['pf_did_rrt'])
    all_pf_did_plane_opt.extend(result['pf_did_plane_opt'])

    clrs.append(result['clearance'])
    reached_all.append(result['reached_goal'])

n_failed = 0
for i, (clr, reached) in enumerate(zip(clrs, reached_all)):
    if clr < CLEARANCE_THRESHOLD or not reached:
        print(f"FAILED: MAP {i+1}")
        n_failed += 1

print("\nPER MAP")
print(f"Total min: {np.min(map_total):.4f} s")
print(f"Total max: {np.max(map_total):.4f} s")
print(f"Total average: {np.mean(map_total):.4f} s")
print(f"Total median: {np.median(map_total):.4f} s")
print(f"RRT median: {np.median(map_rrt):.4f} s")
print(f"Plane median: {np.median(map_plane):.4f} s")
print(f"Control pts median: {np.median(map_q):.4f} s")

print("\nPER FRAME")
n_frames = len(all_pf_t_total)
n_rrt_replans = int(sum(all_pf_did_rrt))
n_plane_opts = int(sum(all_pf_did_plane_opt))

rrt_times_when_run = [t for t, d in zip(all_pf_t_rrt, all_pf_did_rrt) if d]
plane_opt_times_when_run = [t for t, d in zip(all_pf_t_plane_opt, all_pf_did_plane_opt) if d]

print(f"Total time median: {np.median(all_pf_t_total)*1000:.4f} ms")
print(f"RRT {n_rrt_replans}/{n_frames} median: {np.median(rrt_times_when_run)*1000:.4f} ms")
print(f"Planes {n_plane_opts}/{n_frames} median: {np.median(plane_opt_times_when_run)*1000:.4f} ms")
print(f"Control pts median: {np.median(all_pf_t_q)*1000:.4f} ms")   