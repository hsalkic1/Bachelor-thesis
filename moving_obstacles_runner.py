import numpy as np
import matplotlib
if __name__ != "__main__":
    matplotlib.use('Agg')
import os
from trajectory import run as plan_trajectory
from trajectory import DRONE_V, T_REPLAN

GOAL_THRESHOLD = 10.0
MAX_STEPS = 150
CLEARANCE_THRESHOLD = 0.1

def _obs_unpack(obs):
    x = float(obs[0])
    y = float(obs[1])
    z = float(obs[2])
    w = float(obs[3])
    h = float(obs[4])
    d = float(obs[5])
    vx = float(obs[6]) if len(obs) > 6 else 0.0
    vy = float(obs[7]) if len(obs) > 7 else 0.0
    vz = float(obs[8]) if len(obs) > 8 else 0.0
    return x, y, z, w, h, d, vx, vy, vz


#_____ HELPERI
def segment_clearance(sx, sy, sz, obs_list, dt=0.0):
    distances = np.full(len(sx), np.inf)
    zeros = np.zeros(len(sx))
    for obs in obs_list:
        ox, oy, oz, ow, oh, od, vx, vy, vz = _obs_unpack(obs)
        dx = np.maximum.reduce([ox-sx, zeros, sx-(ox+ow)])
        dy = np.maximum.reduce([oy-sy, zeros, sy-(oy+oh)])
        dz = np.maximum.reduce([oz-sz, zeros, sz-(oz+od)])
        distances = np.minimum(distances, np.sqrt(dx**2 + dy**2 + dz**2))
        if dt > 0:
            ex, ey, ez = ox+vx*dt, oy+vy*dt, oz+vz*dt
            dx2 = np.maximum.reduce([ex-sx, zeros, sx-(ex+ow)])
            dy2 = np.maximum.reduce([ey-sy, zeros, sy-(ey+oh)])
            dz2 = np.maximum.reduce([ez-sz, zeros, sz-(ez+od)])
            distances = np.minimum(distances, np.sqrt(dx2**2 + dy2**2 + dz2**2))
    return float(np.min(distances))


def arc_length_u(spline_x, spline_y, spline_z, distance, n_pts=3000):
    u_all = np.linspace(0, 1, n_pts)
    xs, ys, zs = spline_x(u_all), spline_y(u_all), spline_z(u_all)
    segs = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2 + np.diff(zs)**2)
    cum = np.concatenate([[0.0], np.cumsum(segs)])
    if cum[-1] <= distance:
        return 1.0
    idx = int(np.searchsorted(cum, distance))
    idx = max(1, min(idx, n_pts-1))
    t = (distance - cum[idx-1]) / (cum[idx] - cum[idx-1] + 1e-12)
    return float(u_all[idx-1] + t*(u_all[idx] - u_all[idx-1]))


def make_frame(drone, obs, plan_x, plan_y, plan_z, seg_x, seg_y, seg_z, rrt_path=None):
    tx = np.concatenate(seg_x) if seg_x else np.array([drone[0]])
    ty = np.concatenate(seg_y) if seg_y else np.array([drone[1]])
    tz = np.concatenate(seg_z) if seg_z else np.array([drone[2]])

    rrt_x = rrt_path[:, 0] if rrt_path is not None else None
    rrt_y = rrt_path[:, 1] if rrt_path is not None else None
    rrt_z = rrt_path[:, 2] if rrt_path is not None else None

    return dict(drone=drone, obs=list(obs),
                plan_x=plan_x, plan_y=plan_y, plan_z=plan_z,
                trail_x=tx, trail_y=ty, trail_z=tz,
                rrt_x=rrt_x, rrt_y=rrt_y, rrt_z=rrt_z)

#_____ MAIN
def main(number, obstacles, start_pos, goal_pos, map_size):

    pf_t_rrt = []
    pf_t_plane = []
    pf_t_plane_opt = []
    pf_t_q = []
    pf_t_total = []
    pf_did_rrt = []
    pf_did_plane_opt = []

    obs_data = list(obstacles)
    dist_per_step = DRONE_V * T_REPLAN
    current = tuple(start_pos)
    seg_x, seg_y, seg_z = [], [], []
    frames = []
    step_clears = []
    cached_rrt = None
    cached_ctrl_pts = None
    clrs = []
    new_rrt = None

    step = 0
    reached = False
    while True:
        step += 1
        dist = float(np.sqrt((current[0]-goal_pos[0])**2 + (current[1]-goal_pos[1])**2 + (current[2]-goal_pos[2])**2))
        obs_static = [(o[0], o[1], o[2], o[3], o[4], o[5]) for o in obs_data]

        if dist < GOAL_THRESHOLD:
            seg_x.append(np.array([current[0], goal_pos[0]]))
            seg_y.append(np.array([current[1], goal_pos[1]]))
            seg_z.append(np.array([current[2], goal_pos[2]]))
            frames.append(make_frame(goal_pos, obs_static, None, None, None, seg_x, seg_y, seg_z, new_rrt))
            print(f"\nFINISHED: Goal reached at step {step}  (dist={dist:.1f})")
            reached = True
            break

        if step > MAX_STEPS:
            print(f"\nFAILED: MAX_STEPS ({MAX_STEPS}) reached")
            break

        if __name__ == "__main__":
            print(f"Step {step:3d}  pos=({current[0]:6.1f},{current[1]:6.1f},{current[2]:6.1f})")

        pos_cl = segment_clearance(np.array([current[0]]), np.array([current[1]]), np.array([current[2]]),
                                   obs_data, T_REPLAN)
        if pos_cl < CLEARANCE_THRESHOLD or cached_rrt is None:
            rrt_override = None
        else:
            rrt_override = cached_rrt

        N_RETRIES = 5
        best = None
        frame_t_rrt = 0.0
        frame_t_plane = 0.0
        frame_t_plane_opt = 0.0
        frame_t_q = 0.0
        frame_t_total = 0.0
        frame_did_rrt = False
        frame_did_plane_opt = False
        for attempt in range(N_RETRIES):
            override = rrt_override if attempt == 0 else None
            ctrl = cached_ctrl_pts if attempt <= 3 else None
            result = plan_trajectory(obs_data, current, goal_pos, map_size,
                                     override, prev_ctrl_pts=ctrl, dt=T_REPLAN)
            mc, t_rrt_r, t_pl_r, t_po_r, t_q_r, t_tot_r, sx_t, sy_t, sz_t, rrt_t, did_r, did_po = result
            frame_t_rrt += t_rrt_r
            frame_t_plane += t_pl_r
            frame_t_plane_opt += t_po_r
            frame_t_q += t_q_r
            frame_t_total += t_tot_r
            frame_did_rrt = frame_did_rrt or did_r
            frame_did_plane_opt = frame_did_plane_opt or did_po
            if mc is None:
                continue
            if best is None or mc > best[0]:
                best = result
            if mc >= CLEARANCE_THRESHOLD:
                break

        pf_t_rrt.append(frame_t_rrt)
        pf_t_plane.append(frame_t_plane)
        pf_t_plane_opt.append(frame_t_plane_opt)
        pf_t_q.append(frame_t_q)
        pf_t_total.append(frame_t_total)
        pf_did_rrt.append(frame_did_rrt)
        pf_did_plane_opt.append(frame_did_plane_opt)

        mn_cl, _t_rrt_b, _t_pl_b, _t_po_b, _t_q_b, _t_tot_b, spx, spy, spz, new_rrt, _did_r_b, _did_po_b = best

        if new_rrt is not None:
            cached_rrt = new_rrt

        u_end = arc_length_u(spx, spy, spz, dist_per_step)
        us = np.linspace(0.0, u_end, 250)
        sx, sy, sz = spx(us), spy(us), spz(us)

        step_cl = segment_clearance(sx, sy, sz, obs_data, T_REPLAN)
        seg_x.append(sx)
        seg_y.append(sy)
        seg_z.append(sz)
        step_clears.append(step_cl)
        clrs.append(step_cl)
        current = (float(sx[-1]), float(sy[-1]), float(sz[-1]))

        if step_cl < CLEARANCE_THRESHOLD:
            print(f"  !! Step {step}: min clearance = {step_cl:.4f} at pos ({current[0]:.1f}, {current[1]:.1f}, {current[2]:.1f})")
            cached_rrt = None

        h_step = float(spx.t[spx.k + 1] - spx.t[spx.k])
        vel = np.array([float(spx.derivative()(u_end)), float(spy.derivative()(u_end)), float(spz.derivative()(u_end))])
        acc = np.array([float(spx.derivative(2)(u_end)), float(spy.derivative(2)(u_end)), float(spz.derivative(2)(u_end))])
        pos = np.array([current[0], current[1], current[2]])
        cached_ctrl_pts = np.array([
            pos,
            pos + (h_step/3.0) * vel,
            pos + (2.0*h_step/3.0) * vel + (h_step*h_step/6.0) * acc
        ])

        u_f = np.linspace(0, 1, 400)
        frames.append(make_frame(current, obs_static, spx(u_f), spy(u_f), spz(u_f), seg_x, seg_y, seg_z, new_rrt))
        obs_data = [
            (float(np.clip(x+vx*T_REPLAN, 0, map_size[0]-w)),
             float(np.clip(y+vy*T_REPLAN, 0, map_size[1]-h)),
             float(np.clip(z+vz*T_REPLAN, 0, map_size[2]-d)),
             w, h, d, vx, vy, vz)
            for x, y, z, w, h, d, vx, vy, vz in obs_data
        ]


    if __name__ == "__main__":
        print(f"\nTotal steps: {step}")
        print(f"\nTotal time: {sum(pf_t_total):.3f}s")

    _save_anim(number, frames, map_size, start_pos, goal_pos)
    m_clr = float(np.min(clrs)) if clrs else float('inf')
    return dict(
        reached_goal=reached,
        steps=step,
        clearance=m_clr,
        pf_t_rrt=pf_t_rrt,
        pf_t_plane=pf_t_plane,
        pf_t_plane_opt=pf_t_plane_opt,
        pf_t_q=pf_t_q,
        pf_t_total=pf_t_total,
        pf_did_rrt=pf_did_rrt,
        pf_did_plane_opt=pf_did_plane_opt,
    )


#_____ PLOT
def _box_mesh(ox, oy, oz, w, h, d, color='red', opacity=0.25):
    import plotly.graph_objects as go
    x = [ox, ox+w, ox+w, ox, ox, ox+w, ox+w, ox]
    y = [oy, oy, oy+h, oy+h, oy, oy, oy+h, oy+h]
    z = [oz, oz, oz, oz, oz+d, oz+d, oz+d, oz+d]
    i = [0,0,0,4,4,1,1,2,2,3,3,0]
    j = [1,2,4,5,7,5,6,6,7,7,4,3]
    k = [2,3,7,7,3,6,2,7,3,4,0,7]
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=opacity, flatshading=True, showlegend=False)


def _save_anim(number, frames, map_size, start, goal):
    if not frames:
        return
    import plotly.graph_objects as go

    n_obs = len(frames[0]['obs']) if frames else 0

    def line3d(xs, ys, zs, color, width, dash=None, name=None, showlegend=True):
        kwargs = dict(x=xs, y=ys, z=zs, mode='lines',
                      line=dict(color=color, width=width, dash=dash) if dash else dict(color=color, width=width),
                      name=name, showlegend=showlegend)
        return go.Scatter3d(**kwargs)

    def scatter3d(xs, ys, zs, color, size, symbol='circle', name=None, showlegend=True):
        return go.Scatter3d(x=xs, y=ys, z=zs, mode='markers',
                            marker=dict(color=color, size=size, symbol=symbol),
                            name=name, showlegend=showlegend)

    def make_traces(fr):
        traces = []
        traces.append(line3d(fr['trail_x'], fr['trail_y'], fr['trail_z'], 'blue', 4, name='Flown path'))
        if fr['plan_x'] is not None:
            traces.append(line3d(fr['plan_x'], fr['plan_y'], fr['plan_z'], 'cyan', 2, dash='dash', name='Current plan'))
        else:
            traces.append(line3d([], [], [], 'cyan', 2, dash='dash', name='Current plan'))
        if fr['rrt_x'] is not None:
            traces.append(line3d(fr['rrt_x'], fr['rrt_y'], fr['rrt_z'], 'green', 2, name='RRT path'))
        else:
            traces.append(line3d([], [], [], 'green', 2, name='RRT path'))
        dx, dy, dz = fr['drone']
        traces.append(scatter3d([dx], [dy], [dz], 'navy', 6, name='Drone'))
        for ox, oy, oz, ow, oh, od in fr['obs']:
            traces.append(_box_mesh(ox, oy, oz, ow, oh, od, color='red', opacity=0.25))
        while len(traces) < 4 + n_obs:
            traces.append(_box_mesh(0, 0, 0, 0, 0, 0, color='red', opacity=0.0))
        return traces

    init_traces = make_traces(frames[0])
    init_traces.append(scatter3d([start[0]], [start[1]], [start[2]], 'green', 6, symbol='circle', name='Start'))
    init_traces.append(scatter3d([goal[0]], [goal[1]], [goal[2]], 'green', 8, symbol='diamond', name='Goal'))

    plotly_frames = []
    for idx, fr in enumerate(frames):
        plotly_frames.append(go.Frame(
            data=make_traces(fr),
            name=str(idx),
            traces=list(range(4 + n_obs)),
        ))

    fig = go.Figure(data=init_traces, frames=plotly_frames)

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, map_size[0]], title='X'),
            yaxis=dict(range=[0, map_size[1]], title='Y'),
            zaxis=dict(range=[0, map_size[2]], title='Z'),
            aspectmode='cube',
        ),
        title=f"Trajectory {number}",
        updatemenus=[dict(
            type='buttons', showactive=False,
            y=0, x=0.05, xanchor='left', yanchor='top',
            buttons=[
                dict(label='Play', method='animate',
                     args=[None, dict(frame=dict(duration=300, redraw=True), fromcurrent=True)]),
                dict(label='Pause', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]),
            ],
        )],
        sliders=[dict(
            active=0, y=0, x=0.15, len=0.8,
            currentvalue=dict(prefix='Step: '),
            steps=[dict(method='animate',
                        args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode='immediate')],
                        label=str(i)) for i in range(len(frames))],
        )],
    )

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'trajectory_{number}.html')
    fig.write_html(out_path, include_plotlyjs='cdn', auto_play=False)
    if __name__ == "__main__":
        print(f"HTML saved to {out_path}")
        fig.show()
    else:
        print(f"HTML saved to {out_path}")


if __name__ == "__main__":
    obstacles = [
        (40, 40, 40, 40, 40, 40, 2.0, 1.0, 0.5),
        (120, 120, 120, 40, 40, 40, -1.0, -2.0, -0.5),
    ]
    start_pos = (20.0, 20.0, 20.0)
    goal_pos = (180.0, 180.0, 180.0)
    map_size = (200, 200, 200)
    main(0, obstacles, start_pos, goal_pos, map_size)
