import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.optimize import minimize
import time
import nlopt
import scipy.sparse as sp
import cvxpy as cp
import math


DRONE_V=25.0
T_REPLAN=0.5
dist_weight_factor=1.0
jerk_weight_factor=1.0
k=3
SPACE=10
CLEARANCE_THRESHOLD=0.1
USE_ELLIPSE=True
USE_FILTER=True


def _obs_unpack(obs, dt):
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

def _obs_box_corners(obs, dt):
    ox, oy, oz, w, h, dbox, vx, vy, vz = _obs_unpack(obs, dt)
    ex, ey, ez = ox+vx*dt, oy+vy*dt, oz+vz*dt
    return [
        (ox,oy,oz),(ox+w,oy,oz),(ox,oy+h,oz),(ox+w,oy+h,oz),
        (ox,oy,oz+dbox),(ox+w,oy,oz+dbox),(ox,oy+h,oz+dbox),(ox+w,oy+h,oz+dbox),
        (ex,ey,ez),(ex+w,ey,ez),(ex,ey+h,ez),(ex+w,ey+h,ez),
        (ex,ey,ez+dbox),(ex+w,ey,ez+dbox),(ex,ey+h,ez+dbox),(ex+w,ey+h,ez+dbox),
    ]

def _densify_path(pts, space):
    out = [pts[0]]
    for i in range(len(pts)-1):
        p1, p2 = pts[i], pts[i+1]
        d = np.linalg.norm(p2-p1)
        n = max(1, int(d//space))
        for j in range(1, n+1):
            out.append(p1 + (p2-p1)*(j/n))
    return np.array(out)

def run(obstacles, start_pos, goal_pos, map_size, rrt_path_override=None, prev_ctrl_pts=None, dt=0.0):

    #_____ RRT
    class Node:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            self.parent = None

    def _bounds_list(obs_list):
        b = []
        for o in obs_list:
            x, y, z, w, h, d, vx, vy, vz = _obs_unpack(o, dt)
            b.append((x, x+w, y, y+h, z, z+d))
            if vx != 0 or vy != 0 or vz != 0:
                ex = max(0.0, min(x+vx*dt, map_size[0]-w))
                ey = max(0.0, min(y+vy*dt, map_size[1]-h))
                ez = max(0.0, min(z+vz*dt, map_size[2]-d))
                b.append((ex, ex+w, ey, ey+h, ez, ez+d))
        return b

    def _in_box(px, py, pz, bnds):
        for x1, x2, y1, y2, z1, z2 in bnds:
            if x1 <= px <= x2 and y1 <= py <= y2 and z1 <= pz <= z2:
                return True
        return False

    def rrt(start, goal, obstacles, map_size, checkpoints=20):
        tree = [Node(start[0], start[1], start[2])]
        start_bounds = [(o[0], o[0]+o[3], o[1], o[1]+o[4], o[2], o[2]+o[5]) for o in obstacles]
        obs_bounds = _bounds_list(obstacles)

        if _in_box(start[0], start[1], start[2], start_bounds):
            return None

        map_w, map_h, map_d = map_size
        gx, gy, gz = goal
        for _ in range(10000):
            if np.random.rand() > 0.01:
                rx = np.random.rand()*map_w
                ry = np.random.rand()*map_h
                rz = np.random.rand()*map_d
            else:
                rx, ry, rz = gx, gy, gz

            closest = tree[0]
            min_d = float('inf')
            for n in tree:
                dd = (rx-n.x)**2 + (ry-n.y)**2 + (rz-n.z)**2
                if dd < min_d:
                    min_d = dd
                    closest = n

            dxr = rx-closest.x
            dyr = ry-closest.y
            dzr = rz-closest.z
            dn = math.sqrt(dxr*dxr + dyr*dyr + dzr*dzr)
            if dn < 1e-9:
                continue
            nx = closest.x + SPACE*dxr/dn
            ny = closest.y + SPACE*dyr/dn
            nz = closest.z + SPACE*dzr/dn

            if nx<0 or ny<0 or nz<0 or nx>map_w or ny>map_h or nz>map_d:
                continue
            safe = True
            for t in np.linspace(0, 1, checkpoints):
                if _in_box(closest.x + t*(nx-closest.x),
                           closest.y + t*(ny-closest.y),
                           closest.z + t*(nz-closest.z), obs_bounds):
                    safe = False
                    break
            if not safe:
                continue

            new_node = Node(nx, ny, nz)
            new_node.parent = closest
            tree.append(new_node)
            if (gx-nx)**2 + (gy-ny)**2 + (gz-nz)**2 < SPACE*SPACE:
                fn = Node(gx, gy, gz)
                fn.parent = new_node
                tree.append(fn)
                return tree
        return None

    def prune_path(path_nodes, obstacles):
        pts = np.array([[n.x, n.y, n.z] for n in path_nodes])
        obs_bounds = _bounds_list(obstacles)
        map_w, map_h, map_d = map_size

        def line_clear(p1, p2):
            steps = max(1, int(np.linalg.norm(p2-p1)*2))
            for i in range(steps+1):
                t = i/steps
                px = p1[0] + t*(p2[0]-p1[0])
                py = p1[1] + t*(p2[1]-p1[1])
                pz = p1[2] + t*(p2[2]-p1[2])
                if px<0 or py<0 or pz<0 or px>map_w or py>map_h or pz>map_d:
                    return False
                if _in_box(px, py, pz, obs_bounds):
                    return False
            return True

        pruned = [pts[0]]
        i = 0
        while i < len(pts)-1:
            best = i+1
            for j in range(len(pts)-1, i, -1):
                if line_clear(pts[i], pts[j]):
                    best = j
                    break
            pruned.append(pts[best])
            i = best
        return np.array(pruned)

    def rrt_to_path(tree, obstacles):
        last = tree[-1]
        raw = [last]
        curr = last.parent
        while curr is not None:
            raw.append(curr)
            curr = curr.parent
        raw.reverse()
        return raw, _densify_path(prune_path(raw, obstacles), SPACE)

    t_rrt = 0.0
    t_plane_gen = 0.0
    t_plane_opt = 0.0
    t_q = 0.0
    did_rrt = False
    did_plane_opt = False
    t_start_total = time.perf_counter()
    _t_rrt_start = time.perf_counter()

    if rrt_path_override is not None:
        full = np.array(rrt_path_override, dtype=float)
        closest = int(np.argmin(np.linalg.norm(full - np.array(start_pos), axis=1)))
        rrt_path = full[closest:].copy()
        rrt_path[0] = start_pos
        while len(rrt_path) < 4:
            rrt_path = np.vstack([rrt_path, (rrt_path[-1] + np.array(goal_pos)) / 2])
        rrt_tree = None
        raw_path = []
    else:
        did_rrt = True
        rrt_tree = rrt(start_pos, goal_pos, obstacles, map_size)
        if rrt_tree is None:
            print("RRT timeout - using straight-line seed")
            sp_pt = np.array(start_pos, dtype=float)
            gp = np.array(goal_pos, dtype=float)
            n_pts = max(4, int(float(np.linalg.norm(gp - sp_pt)) // SPACE) + 1)
            rrt_path = np.array([sp_pt + t*(gp - sp_pt) for t in np.linspace(0.0, 1.0, n_pts)])
            raw_path = []
        else:
            raw_path, rrt_path = rrt_to_path(rrt_tree, obstacles)

    n_rrt = len(rrt_path)
    if n_rrt < 2:
        sp_pt = np.array(start_pos, dtype=float)
        gp = np.array(goal_pos, dtype=float)
        rrt_path = np.array([sp_pt, sp_pt + (gp-sp_pt)*0.33, sp_pt + (gp-sp_pt)*0.66, gp])
        n_rrt = len(rrt_path)
    while n_rrt < 4:
        rrt_path = np.insert(rrt_path, -1, (rrt_path[-2] + rrt_path[-1]) / 2, axis=0)
        n_rrt = len(rrt_path)

    t_rrt += time.perf_counter() - _t_rrt_start

    #_____ Q INIT
    N = n_rrt + 4
    N_free = n_rrt - 2
    n_segments = N - 3

    init_Q = np.zeros((N, 3))
    init_Q[0:3] = np.array(prev_ctrl_pts) if prev_ctrl_pts is not None else rrt_path[0]
    init_Q[3:3+N_free] = rrt_path[1:-1]
    init_Q[3+N_free:] = rrt_path[-1]

    q_fixed_start = init_Q[0:3].copy()
    q_fixed_end = init_Q[-3:].copy()

    #_____ TANGENT PLANE INIT
    def get_tangent_plane(control_points, obs):
        ox, oy, oz, w, h, dbox, vx, vy, vz = _obs_unpack(obs, dt)
        r_static = math.sqrt((w/2)**2 + (h/2)**2 + (dbox/2)**2)
        c1 = np.array([ox+w/2, oy+h/2, oz+dbox/2])
        c2 = c1 + np.array([vx*dt, vy*dt, vz*dt])
        obs_center = (c1 + c2) / 2.0
        motion = c2 - c1
        motion_len = float(np.linalg.norm(motion))
        obs_corners = np.array(_obs_box_corners(obs, dt))
        seg_center = control_points.mean(axis=0)

        if USE_ELLIPSE:
            if motion_len < 1e-9:
                axis_u = np.array([1.0, 0.0, 0.0])
                axis_v = np.array([0.0, 1.0, 0.0])
                axis_w = np.array([0.0, 0.0, 1.0])
                a_e = b_e = c_e = 1.1*r_static
            else:
                axis_u = motion / motion_len
                tmp = np.array([0.0, 0.0, 1.0]) if abs(axis_u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
                axis_v = np.cross(axis_u, tmp)
                axis_v /= (np.linalg.norm(axis_v) + 1e-12)
                axis_w = np.cross(axis_u, axis_v)
                a_e = motion_len/2.0 + 1.1*r_static
                b_e = c_e = 1.1*r_static

            diff = seg_center - obs_center
            pu = float(np.dot(diff, axis_u))
            pv = float(np.dot(diff, axis_v))
            pw = float(np.dot(diff, axis_w))
            denom = float(math.sqrt(a_e**2*pu**2 + b_e**2*pv**2 + c_e**2*pw**2)) + 1e-12
            N_vec = pu*axis_u + pv*axis_v + pw*axis_w
            a, b, c = float(-N_vec[0]), float(-N_vec[1]), float(-N_vec[2])
            D = float(N_vec[0]*obs_center[0] + N_vec[1]*obs_center[1] + N_vec[2]*obs_center[2] + denom)
        else:
            obs_r = float(motion_len/2.0 + 1.1*r_static)
            vec = obs_center - seg_center
            normal = vec / (float(np.linalg.norm(vec)) + 1e-9)
            a, b, c = float(normal[0]), float(normal[1]), float(normal[2])
            cp_pt = obs_center - normal*obs_r
            D = float(-(a*cp_pt[0] + b*cp_pt[1] + c*cp_pt[2]))

        min_obs = min(a*cx + b*cy + c*cz + D for cx, cy, cz in obs_corners)

        if min_obs < 1e-6:
            max_seg = max(a*p[0] + b*p[1] + c*p[2] for p in control_points)
            min_obs = min(a*cx + b*cy + c*cz for cx, cy, cz in obs_corners)
            if max_seg > min_obs:
                a, b, c = -a, -b, -c
                max_seg = max(a*p[0] + b*p[1] + c*p[2] for p in control_points)
                min_obs = min(a*cx + b*cy + c*cz for cx, cy, cz in obs_corners)
            gap = min_obs - max_seg
            if gap < 1e-6:
                return None
            alpha = 2.0 / gap
            D = -(max_seg + min_obs) / 2.0
            return a*alpha, b*alpha, c*alpha, D*alpha
        alpha = 1.0 / min_obs
        return a*alpha, b*alpha, c*alpha, D*alpha

    #_____ TANGENT PLANE OP
    def optimize_single_plane(a0, b0, c0, D0, seg_pts, obs_corners):
        lb = np.array([-100.0, -100.0, -100.0, -1000.0])
        ub = np.array([ 100.0,  100.0,  100.0,  1000.0])
        x0 = np.array([a0, b0, c0, D0], dtype=float)
        if not np.all(np.isfinite(x0)):
            x0 = np.array([0.1, 0.1, 0.1, -10.0], dtype=float)
        x0 = np.clip(x0, lb, ub)
        last_x = [x0.copy()]

        def obj(x, grad, lx=last_x):
            lx[0] = x.copy()
            if grad.size > 0:
                grad[:] = [2*x[0], 2*x[1], 2*x[2], 0.0]
            return float(x[0]**2 + x[1]**2 + x[2]**2)

        def cc(x, grad, pt):
            if grad.size > 0:
                grad[:] = [pt[0], pt[1], pt[2], 1.0]
            return float(1 + x[0]*pt[0] + x[1]*pt[1] + x[2]*pt[2] + x[3])

        def oc(x, grad, cr):
            if grad.size > 0:
                grad[:] = [-cr[0], -cr[1], -cr[2], -1.0]
            return float(1 - x[0]*cr[0] - x[1]*cr[1] - x[2]*cr[2] - x[3])

        opt = nlopt.opt(nlopt.LD_SLSQP, 4)
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)
        opt.set_min_objective(obj)
        opt.set_xtol_rel(1e-3)
        opt.set_ftol_rel(1e-3)
        for pt in seg_pts:
            opt.add_inequality_constraint(lambda x, g, p=pt: cc(x, g, p), 1e-3)
        for cr in obs_corners:
            opt.add_inequality_constraint(lambda x, g, c=cr: oc(x, g, c), 1e-3)
        try:
            return tuple(opt.optimize(x0))
        except nlopt.RoundoffLimited:
            return tuple(last_x[0])
        except Exception as e:
            print(f"UNHANDELED (planes): {type(e).__name__}: {e}")
            return (a0, b0, c0, D0)

    def build_planes(optimize=False, return_active=False):
        A, B, C, Dl, QI = [], [], [], [], []
        active = []

        for seg_i in range(n_segments):
            ctrl_pts = init_Q[seg_i:seg_i+4]
            seg_c = ctrl_pts.mean(axis=0)
            seg_rad = float(max(np.linalg.norm(p - seg_c) for p in ctrl_pts))
            for obs in obstacles:
                ox, oy, oz, w, h, dbox, ovx, ovy, ovz = _obs_unpack(obs, dt)
                r_static = math.sqrt((w/2)**2 + (h/2)**2 + (dbox/2)**2)
                cxm = ox + ovx*dt*0.5 + w/2
                cym = oy + ovy*dt*0.5 + h/2
                czm = oz + ovz*dt*0.5 + dbox/2
                r_swept = r_static*1.2 + math.sqrt((ovx*dt)**2 + (ovy*dt)**2 + (ovz*dt)**2)/2.0
                d_seg_obs = math.sqrt((cxm-seg_c[0])**2 + (cym-seg_c[1])**2 + (czm-seg_c[2])**2)
                if USE_FILTER and d_seg_obs > 3.0*seg_rad + r_swept + max(w, h, dbox) + 10.0:
                    continue
                obs_corners = _obs_box_corners(obs, dt)
                plane = get_tangent_plane(ctrl_pts, obs)
                if plane is None:
                    continue
                a, b, c, D = plane
                if optimize:
                    a, b, c, D = optimize_single_plane(a, b, c, D, ctrl_pts, obs_corners)
                for q_i in range(4):
                    idx = seg_i + q_i
                    if 3 <= idx < N - 3:
                        A.append(a)
                        B.append(b)
                        C.append(c)
                        Dl.append(D)
                        QI.append(idx)
                active.append((seg_i, obs_corners, a, b, c, D))

        if return_active:
            return A, B, C, Dl, QI, active
        return A, B, C, Dl, QI

    #____ CONTROL POINTS OPT (CONVEX)
    def solve_nodes(A_list, B_list, C_list, D_list, Q_idx_list):
        M = len(A_list)
        Q_free_var = cp.Variable((N_free, 3))
        Q = cp.vstack([q_fixed_start, Q_free_var, q_fixed_end])

        eval_pts = (Q[2:2+N_free] + 4*Q[3:3+N_free] + Q[4:4+N_free]) / 6.0
        cost_dist = cp.sum_squares(eval_pts - rrt_path[1:-1])
        jerk = (-Q[0:n_segments] + 3*Q[1:n_segments+1] - 3*Q[2:n_segments+2] + Q[3:n_segments+3])
        cost_jerk = cp.sum_squares(jerk)
        total_cost = dist_weight_factor*cost_dist + jerk_weight_factor*cost_jerk

        constraints = [
            Q_free_var >= 0,
            Q_free_var[:, 0] <= map_size[0],
            Q_free_var[:, 1] <= map_size[1],
            Q_free_var[:, 2] <= map_size[2],
        ]

        if M > 0:
            A_v = np.array(A_list)
            B_v = np.array(B_list)
            C_v = np.array(C_list)
            D_v = np.array(D_list)
            H = sp.csr_matrix((np.ones(M), (np.arange(M), Q_idx_list)), shape=(M, N))
            Q_sel = H @ Q
            constraints.append(
                cp.multiply(Q_sel[:, 0], A_v) + cp.multiply(Q_sel[:, 1], B_v) +
                cp.multiply(Q_sel[:, 2], C_v) + D_v + 0.01 <= 0
            )

        try:
            prob = cp.Problem(cp.Minimize(total_cost), constraints)
            try:
                prob.solve(solver=cp.CLARABEL)
            except Exception:
                prob.solve(solver=cp.OSQP, eps_rel=1e-3, eps_abs=1e-3, max_iter=500000, warm_start=True)
            val = Q_free_var.value
            if val is not None and prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.USER_LIMIT]:
                return val, prob.status
            return None, prob.status
        except Exception as e:
            print(f"UNHANDELED (nodes): {type(e).__name__}: {e}")
            return None, "exception"

    #____ JOINT NLP FALLBACK
    def joint_nonlinear():
        _, _, _, _, _, active_init = build_planes(optimize=False, return_active=True)
        active = []
        for (seg_i, obs_corners, a0, b0, c0, D0) in active_init:
            ctrl_pts = init_Q[seg_i:seg_i+4]
            a, b, c, D = optimize_single_plane(a0, b0, c0, D0, ctrl_pts, obs_corners)
            active.append((seg_i, obs_corners, a, b, c, D))

        n_pl = len(active)
        n_q_vars = N_free * 3
        n_vars = n_q_vars + n_pl * 4

        x0_parts = [init_Q[3:3+N_free].flatten()]
        for (_, _, a, b, c, D) in active:
            x0_parts.append(np.array([a, b, c, D]))
        x0 = np.concatenate(x0_parts) if active else init_Q[3:3+N_free].flatten()

        target_pts = rrt_path[1:-1]
        eps_pl = 0.01

        def obj_and_grad(x):
            Qf = x[:n_q_vars].reshape(N_free, 3)
            Q = np.vstack([q_fixed_start, Qf, q_fixed_end])
            ev = (Q[2:2+N_free] + 4*Q[3:3+N_free] + Q[4:4+N_free]) / 6.0
            diff = ev - target_pts
            jk = -Q[0:n_segments] + 3*Q[1:n_segments+1] - 3*Q[2:n_segments+2] + Q[3:n_segments+3]
            obj = dist_weight_factor*float(np.sum(diff*diff)) + jerk_weight_factor*float(np.sum(jk*jk))

            grad_Q = np.zeros_like(Q)
            coef = (2.0/6.0) * dist_weight_factor * diff
            grad_Q[2:2+N_free] += coef
            grad_Q[3:3+N_free] += 4*coef
            grad_Q[4:4+N_free] += coef
            j_coef = 2.0 * jerk_weight_factor * jk
            grad_Q[0:n_segments] -= j_coef
            grad_Q[1:n_segments+1] += 3*j_coef
            grad_Q[2:n_segments+2] -= 3*j_coef
            grad_Q[3:n_segments+3] += j_coef

            g = np.zeros(n_vars)
            g[:n_q_vars] = grad_Q[3:3+N_free].flatten()
            return obj, g

        bounds = []
        for _ in range(N_free):
            bounds.extend([(0.0, float(map_size[0])), (0.0, float(map_size[1])), (0.0, float(map_size[2]))])
        bounds.extend([(-100.0, 100.0), (-100.0, 100.0), (-100.0, 100.0), (-1000.0, 1000.0)] * n_pl)

        constraints = []
        for pi, (seg_i, obs_corners, *_) in enumerate(active):
            plane_off = n_q_vars + pi*4
            for q_i in range(4):
                idx = seg_i + q_i
                if not (3 <= idx < N - 3):
                    continue
                qx_off = (idx - 3) * 3

                def make_cc(plane_off=plane_off, qx_off=qx_off, eps=eps_pl):
                    def f(x):
                        a, b, c, D = x[plane_off:plane_off+4]
                        qx, qy, qz = x[qx_off:qx_off+3]
                        return -(a*qx + b*qy + c*qz + D + eps)
                    def j(x):
                        a, b, c, D = x[plane_off:plane_off+4]
                        qx, qy, qz = x[qx_off:qx_off+3]
                        gj = np.zeros(n_vars)
                        gj[qx_off:qx_off+3] = [-a, -b, -c]
                        gj[plane_off:plane_off+4] = [-qx, -qy, -qz, -1.0]
                        return gj
                    return f, j
                f_cc, j_cc = make_cc()
                constraints.append({'type': 'ineq', 'fun': f_cc, 'jac': j_cc})

            for cr in obs_corners:
                def make_oc(plane_off=plane_off, cr=cr, eps=eps_pl):
                    crx, cry, crz = float(cr[0]), float(cr[1]), float(cr[2])
                    def f(x):
                        a, b, c, D = x[plane_off:plane_off+4]
                        return a*crx + b*cry + c*crz + D - eps
                    def j(x):
                        gj = np.zeros(n_vars)
                        gj[plane_off:plane_off+4] = [crx, cry, crz, 1.0]
                        return gj
                    return f, j
                f_oc, j_oc = make_oc()
                constraints.append({'type': 'ineq', 'fun': f_oc, 'jac': j_oc})

        try:
            res = minimize(obj_and_grad, x0, jac=True, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 100, 'ftol': 1e-3})
            return res.x[:n_q_vars].reshape(N_free, 3), "nlp"
        except Exception as e:
            print(f"UNHANDELED (joint nlp): {type(e).__name__}: {e}")
            return None, "nlp_exception"

    #____ SPLINE EVAL + CLEARANCE
    def get_clearances(sx, sy, sz):
        distances = np.full(len(sx), np.inf)
        zeros = np.zeros(len(sx))
        for obs in obstacles:
            ox, oy, oz, w, h, dbox, vx, vy, vz = _obs_unpack(obs, dt)
            for ox_, oy_, oz_ in [(ox, oy, oz), (ox+vx*dt, oy+vy*dt, oz+vz*dt)]:
                dx = np.maximum.reduce([ox_-sx, zeros, sx-(ox_+w)])
                dy = np.maximum.reduce([oy_-sy, zeros, sy-(oy_+h)])
                dz = np.maximum.reduce([oz_-sz, zeros, sz-(oz_+dbox)])
                distances = np.minimum(distances, np.sqrt(dx**2 + dy**2 + dz**2))
        return distances

    def eval_Q(Q_pts, t_spl_):
        sx = BSpline(t_spl_, Q_pts[:, 0], k)(u_dense)
        sy = BSpline(t_spl_, Q_pts[:, 1], k)(u_dense)
        sz = BSpline(t_spl_, Q_pts[:, 2], k)(u_dense)
        clear = get_clearances(sx, sy, sz)
        return sx, sy, sz, clear, float(np.min(clear))

    #_____ "MAIN"
    _t = time.perf_counter()
    A_list, B_list, C_list, D_list, Q_idx_list = build_planes(optimize=False)
    t_plane_gen += time.perf_counter() - _t

    _t = time.perf_counter()
    opt_q_free, status = solve_nodes(A_list, B_list, C_list, D_list, Q_idx_list)
    t_q += time.perf_counter() - _t

    t_spl = np.concatenate([np.zeros(k), np.linspace(0, 1, N-k+1), np.ones(k)])
    u_dense = np.linspace(0, 1, 1000)

    if opt_q_free is None:
        opt_q_free = init_Q[3:3+N_free]
    opt_Q = np.vstack([q_fixed_start, opt_q_free, q_fixed_end])
    opt_sx, opt_sy, opt_sz, clearances, min_clearance = eval_Q(opt_Q, t_spl)

    if min_clearance < CLEARANCE_THRESHOLD:
        print("DOING PLANE OPTIMISATION")
        did_plane_opt = True
        _t = time.perf_counter()
        A_list, B_list, C_list, D_list, Q_idx_list = build_planes(optimize=True)
        t_plane_opt += time.perf_counter() - _t

        _t = time.perf_counter()
        opt_q_free, status = solve_nodes(A_list, B_list, C_list, D_list, Q_idx_list)
        t_q += time.perf_counter() - _t
        if opt_q_free is None:
            opt_q_free = init_Q[3:3+N_free]
        opt_Q = np.vstack([q_fixed_start, opt_q_free, q_fixed_end])
        opt_sx, opt_sy, opt_sz, clearances, min_clearance = eval_Q(opt_Q, t_spl)

    if min_clearance < CLEARANCE_THRESHOLD:
        print("DOING FRESH RRT FALLBACK")
        did_rrt = True
        _t = time.perf_counter()
        fresh_tree = rrt(start_pos, goal_pos, obstacles, map_size)
        t_rrt += time.perf_counter() - _t
        if fresh_tree is not None:
            _, fp = rrt_to_path(fresh_tree, obstacles)
            while len(fp) < 4:
                fp = np.insert(fp, -1, (fp[-2]+fp[-1])/2, axis=0)

            N_bk, N_free_bk, n_segments_bk = N, N_free, n_segments
            init_Q_bk, rrt_path_bk, t_spl_bk = init_Q, rrt_path, t_spl

            n_rrt_f = len(fp)
            N = n_rrt_f + 4
            N_free = n_rrt_f - 2
            n_segments = N - 3
            init_Q = np.zeros((N, 3))
            init_Q[0:3] = q_fixed_start
            init_Q[3:3+N_free] = fp[1:-1]
            init_Q[3+N_free:] = q_fixed_end
            rrt_path = fp
            t_spl = np.concatenate([np.zeros(k), np.linspace(0, 1, N-k+1), np.ones(k)])

            did_plane_opt = True
            _t = time.perf_counter()
            Af, Bf, Cf, Df_l, QIf = build_planes(optimize=True)
            t_plane_opt += time.perf_counter() - _t

            _t = time.perf_counter()
            qf, stf = solve_nodes(Af, Bf, Cf, Df_l, QIf)
            t_q += time.perf_counter() - _t
            if qf is None:
                qf = init_Q[3:3+N_free]
            opt_Q_f = np.vstack([q_fixed_start, qf, q_fixed_end])
            sx_f, sy_f, sz_f, cl_f, mc_f = eval_Q(opt_Q_f, t_spl)
            t_spl_f_saved = t_spl

            N, N_free, n_segments = N_bk, N_free_bk, n_segments_bk
            init_Q = init_Q_bk
            rrt_path = rrt_path_bk
            t_spl = t_spl_bk

            if mc_f > min_clearance:
                opt_Q, opt_sx, opt_sy, opt_sz = opt_Q_f, sx_f, sy_f, sz_f
                clearances, min_clearance = cl_f, mc_f
                status = stf + "_fresh_rrt"
                rrt_path = fp
                t_spl = t_spl_f_saved

    if min_clearance < CLEARANCE_THRESHOLD:
        print("DOING JOINT NLP FALLBACK")
        did_plane_opt = True
        _t = time.perf_counter()
        qj, stj = joint_nonlinear()
        t_q += time.perf_counter() - _t
        if qj is not None:
            opt_Q_j = np.vstack([q_fixed_start, qj, q_fixed_end])
            sx_j, sy_j, sz_j, cl_j, mc_j = eval_Q(opt_Q_j, t_spl)
            if mc_j > min_clearance:
                opt_Q, opt_sx, opt_sy, opt_sz = opt_Q_j, sx_j, sy_j, sz_j
                clearances, min_clearance = cl_j, mc_j
                status = stj

    t_total = time.perf_counter() - t_start_total
    t_plane = t_plane_gen + t_plane_opt

    if __name__ == "__main__":
        print(f"Status: {status}")
        print(f"Minimum clearance:  {min_clearance:.4f}")
        print(f"Time total:         {t_total:.3f}s")
        print(f"  RRT:              {t_rrt:.3f}s")
        print(f"  Plane gen:        {t_plane_gen:.3f}s")
        print(f"  Plane opt:        {t_plane_opt:.3f}s")
        print(f"  Nodes (OSQP):     {t_q:.3f}s")

    spline_x = BSpline(t_spl, opt_Q[:, 0], k)
    spline_y = BSpline(t_spl, opt_Q[:, 1], k)
    spline_z = BSpline(t_spl, opt_Q[:, 2], k)

    #_____ PLOT
    if __name__ == "__main__":
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter([start_pos[0]], [start_pos[1]], [start_pos[2]], color='black', s=40, marker='o', label='Start')
        ax.scatter([goal_pos[0]], [goal_pos[1]], [goal_pos[2]], color='black', s=40, marker='x', label='Goal')
        ax.set_xlim(0, map_size[0])
        ax.set_ylim(0, map_size[1])
        ax.set_zlim(0, map_size[2])

        for obs in obstacles:
            ox, oy, oz, w, h, dbox, *_ = _obs_unpack(obs, dt)
            ax.bar3d(ox, oy, oz, w, h, dbox, color='red', alpha=0.3)

        for node in (rrt_tree or []):
            if node.parent:
                ax.plot([node.x, node.parent.x], [node.y, node.parent.y], [node.z, node.parent.z],
                        color='gray', linewidth=0.5)

        ax.plot(rrt_path[:, 0], rrt_path[:, 1], rrt_path[:, 2],
                marker='o', color='orange', markersize=4, linewidth=1, label='Pruned RRT path')

        bsx = BSpline(t_spl, init_Q[:, 0], k)(u_dense)
        bsy = BSpline(t_spl, init_Q[:, 1], k)(u_dense)
        bsz = BSpline(t_spl, init_Q[:, 2], k)(u_dense)
        ax.plot(bsx, bsy, bsz, color='black', lw=0.5, label='Init B-Spline')
        ax.plot(opt_sx, opt_sy, opt_sz, color='blue', lw=2, label='Optimized B-Spline')
        ax.scatter(opt_Q[:, 0], opt_Q[:, 1], opt_Q[:, 2], color='blue', s=20, label='Control pts')
        ax.legend(loc='upper left', fontsize=8)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(u_dense, clearances, color='blue')
        ax2.set_xlabel('u')
        ax2.set_ylabel('Clearance')
        ax2.set_title('Clearance')
        plt.show()

    return (min_clearance, t_rrt, t_plane, t_plane_opt, t_q, t_total,
            spline_x, spline_y, spline_z, rrt_path, did_rrt, did_plane_opt)


if __name__ == "__main__":
    obstacles = [(40, 40, 40, 40, 40, 40), (120, 120, 120, 40, 40, 40)]
    start_pos = (10, 10, 10)
    goal_pos = (190, 190, 190)
    map_size = (200, 200, 200)
    run(obstacles, start_pos, goal_pos, map_size)
