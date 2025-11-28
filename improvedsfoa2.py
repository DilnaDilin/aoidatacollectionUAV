#!/usr/bin/env python3
"""
sfw_multi_uav_aoi_weighted.py

SFW-based solver for AoI multi-UAV data collection.
Fitness = weighted combination of:
  - peak AoI (normalized)
  - average AoI (normalized)
  - total UAV travel distance (normalized)
  - Tmax violation penalty (normalized overtime sum * penalty_coeff)

Minimize fitness.

Run: python sfw_multi_uav_aoi_weighted.py
Requires: numpy, matplotlib
"""
import random
import math
import copy
import time
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Problem parameters (tweak as needed)
# ----------------------------
AREA_SIZE = 500
UAV_SPEED = 10.0
T_MAX = 250
R_IOT = 10.0
R_GBS = 30.0
NUM_UAV = 3
NUM_GBS = 4
NUM_IOT = 50
SEED = 42

# SFW hyperparameters (tune to match GA/EIPGA budget)
POP = 30
MAX_ITERS = 100
LEVY_BETA = 1.5
# SFW-specific parameters (following MATLAB variable choices)
C_PARAM = 0.8

# Fitness weights (sum not required; they weight each term)
W_PEAK = 0.6
W_AVG = 0.3
W_DIST = 0.1
W_PEN = 0.1
GB_IMMEDIATE_INSERT_THRESHOLD = 2 * R_GBS  # threshold to insert GB immediately
# Penalty coefficient for Tmax violations (large)
PENALTY_COEFF = 1000.0


# --- AFGMS (Adaptive Fitness-Guided Mutation Strategy) ---
PM_LOW = 0.02     # minimum mutation probability (for best individuals)
PM_HIGH = 0.45    # maximum mutation probability (for poor individuals)
GAMMA = 1.0       # sensitivity exponent
SIGMA_G = 0.04    # Gaussian noise scale
SIGMA_C = 0.08    # Cauchy noise scale (heavy-tailed jumps)
SCALE0 = 1.0      # base mutation scale (will decay over iterations)
EPS = 1e-12

# Encoding dimension: keys + (num_uav-1) continuous breakpoints
DIM = NUM_IOT + max(0, NUM_UAV - 1)

random.seed(SEED)
np.random.seed(SEED)

# ----------------------------
# Utilities: geometry & sim (kept identical to prior implementations)
# ----------------------------
def dist(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return float(np.linalg.norm(a - b))

def travel_time(a, b, V):
    d = dist(a, b)
    return math.ceil(d / V)

def point_on_line_at_distance_from_target(prev_pos, target_pos, radius):
    prev = np.array(prev_pos, dtype=float)
    tgt = np.array(target_pos, dtype=float)
    d = dist(prev, tgt)
    if d <= radius:
        return tuple(prev)
    vec = prev - tgt
    if np.isclose(np.linalg.norm(vec), 0.0):
        return tuple(tgt + np.array([radius, 0.0]))
    unit = vec / np.linalg.norm(vec)
    return tuple(tgt + unit * radius)

def clean_path_remove_consecutive_gbs(path, uav):
    cleaned = []
    for entry in path:
        if cleaned and cleaned[-1][0] == 'gb' and entry[0] == 'gb':
            continue
        cleaned.append(entry)
    if not cleaned:
        cleaned = [('start', None, uav.start), ('end', None, uav.end)]
        return cleaned
    if cleaned[0][0] != 'start':
        cleaned.insert(0, ('start', None, uav.start))
    if cleaned[-1][0] != 'end':
        cleaned.append(('end', None, uav.end))
    return cleaned

# ----------------------------
# Entities
# ----------------------------
class IoT:
    def __init__(self, _id, x, y, t_gen):
        self.id = int(_id)
        self.pos = (float(x), float(y))
        self.t_gen = int(t_gen)

class GBS:
    def __init__(self, _id, x, y):
        self.id = int(_id)
        self.pos = (float(x), float(y))

class UAV:
    def __init__(self, _id, start, end, V, Tmax):
        self.id = int(_id)
        self.start = tuple(np.array(start, dtype=float))
        self.end = tuple(np.array(end, dtype=float))
        self.V = float(V)
        self.Tmax = float(Tmax)
        self.path = [('start', None, self.start), ('end', None, self.end)]

    def copy(self):
        u = UAV(self.id, self.start, self.end, self.V, self.Tmax)
        u.path = copy.deepcopy(self.path)
        return u

# ----------------------------
# Evaluation code (range-aware)
# ----------------------------
def evaluate_uav_path(uav, iot_objs, gb_objs, R_I=R_IOT, R_G=R_GBS):
    V_D = {iot.id: None for iot in iot_objs}
    V_U = {iot.id: None for iot in iot_objs}
    iot_tgen = {iot.id: iot.t_gen for iot in iot_objs}
    cur_pos = np.array(uav.path[0][2], dtype=float)
    t = 0
    total_dist = 0.0
    captured_pending = []

    for typ, idx, wpos in uav.path[1:]:
        wpos = tuple(wpos)
        d = dist(cur_pos, wpos)
        total_dist += d
        tt = travel_time(cur_pos, wpos, uav.V)
        t += tt
        cur_pos = np.array(wpos, dtype=float)

        if typ == 'iot':
            if t < iot_tgen[idx]:
                wait = int(iot_tgen[idx] - t)
                t += wait
            V_D[idx] = int(t)
            captured_pending.append(idx)
        elif typ == 'gb':
            for iid in list(captured_pending):
                V_U[iid] = int(t)
                captured_pending.remove(iid)
        elif typ in ('start', 'end'):
            pass

    if captured_pending:
        nearest_g = min(gb_objs, key=lambda g: dist(cur_pos, g.pos))
        g_w = point_on_line_at_distance_from_target(cur_pos, nearest_g.pos, R_G)
        d = dist(cur_pos, g_w)
        total_dist += d
        tt = travel_time(cur_pos, g_w, uav.V)
        t += tt
        for iid in list(captured_pending):
            V_U[iid] = int(t)
            captured_pending.remove(iid)

    finish_time = int(t)
    feasible = finish_time <= uav.Tmax
    return V_D, V_U, total_dist, finish_time, feasible

def evaluate_all_uavs(uav_list, iot_objs, gb_objs):
    V_D_glob = {iot.id: None for iot in iot_objs}
    V_U_glob = {iot.id: None for iot in iot_objs}
    per_uav_lengths = {}
    per_uav_finish_times = {}
    feasibility = {}
    for u in uav_list:
        V_D, V_U, plen, tfin, feas = evaluate_uav_path(u, iot_objs, gb_objs)
        per_uav_lengths[u.id] = plen
        per_uav_finish_times[u.id] = tfin
        feasibility[u.id] = feas
        for iid in V_D:
            if V_D[iid] is not None and V_D_glob[iid] is None:
                V_D_glob[iid] = V_D[iid]
        for iid in V_U:
            if V_U[iid] is not None and V_U_glob[iid] is None:
                V_U_glob[iid] = V_U[iid]
    return V_D_glob, V_U_glob, per_uav_lengths, per_uav_finish_times, feasibility

def compute_aoi_from_VU(V_U_glob, iot_objs):
    aoi = {}
    aois_list = []
    for iot in iot_objs:
        v = V_U_glob.get(iot.id)
        if v is not None:
            val = int(v - iot.t_gen)
            aoi[iot.id] = val
            aois_list.append(val)
        else:
            aoi[iot.id] = None
    peak = max(aois_list) if aois_list else None
    avg = sum(aois_list) / len(aois_list) if aois_list else None
    coverage = sum(1 for v in aoi.values() if v is not None) / len(iot_objs)
    return aoi, peak, avg, coverage

# ----------------------------
# Build routes from assignment (same as GA/EIPGA)
# ----------------------------
def build_routes_from_assignment(assign: List[int], iot_objs: List[IoT], gb_objs: List[GBS], uav_list: List[UAV], R_I=R_IOT, R_G=R_GBS):
    assigned = {u.id: [] for u in uav_list}
    for idx, gene in enumerate(assign):
        uidx = int(gene)
        u_id = uidx + 1
        if u_id in assigned:
            assigned[u_id].append(iot_objs[idx].id)

    routes = {u.id: [('start', None, u.start)] for u in uav_list}
    iot_map = {iot.id: iot for iot in iot_objs}

    for u in uav_list:
        cur_pos = u.start
        remaining = assigned[u.id][:]
        order = []
        while remaining:
            best = min(remaining, key=lambda iid: dist(cur_pos, iot_map[iid].pos))
            order.append(best)
            cur_pos = iot_map[best].pos
            remaining.remove(best)
        prev_pos = u.start
        for iid in order:
            iobj = iot_map[iid]
            i_wp = point_on_line_at_distance_from_target(prev_pos, iobj.pos, R_I)
            nearest_g = min(gb_objs, key=lambda g: dist(i_wp, g.pos))
            g_wp = point_on_line_at_distance_from_target(i_wp, nearest_g.pos, R_G)
            routes[u.id].append(('iot', iid, i_wp))
            if dist(i_wp, g_wp) < 2 * R_GBS:
                routes[u.id].append(('gb', nearest_g.id, g_wp))
                prev_pos = g_wp
            else:
                prev_pos = i_wp
        routes[u.id].append(('end', None, u.end))
        routes[u.id] = clean_path_remove_consecutive_gbs(routes[u.id], u)

    uav_paths = {}
    for u in uav_list:
        uav_paths[u.id] = routes[u.id]
    return uav_paths

# ----------------------------
# NEW: evaluate_assignment_fitness -> composite weighted score (minimize)
# ----------------------------
def evaluate_assignment_fitness(assign: List[int], iot_objs, gb_objs, base_uavs,
                                w_peak=W_PEAK, w_avg=W_AVG, w_dist=W_DIST, w_pen=W_PEN,
                                penalty_coeff=PENALTY_COEFF, w_penalty=1000.0):
    uav_copies = [u.copy() for u in base_uavs]
    uav_paths = build_routes_from_assignment(assign, iot_objs, gb_objs, uav_copies)
    for u in uav_copies:
        u.path = uav_paths[u.id]

    # fallback uploads
    for u in uav_copies:
        V_D, V_U, plen, tfin, feas = evaluate_uav_path(u, iot_objs, gb_objs)
        pending = [iid for iid in V_D if V_D[iid] is not None and V_U[iid] is None]
        if pending:
            nearest_g = min(gb_objs, key=lambda g: dist(u.path[-2][2], g.pos))
            g_wp = point_on_line_at_distance_from_target(u.path[-2][2], nearest_g.pos, R_G)
            u.path = u.path[:-1] + [('gb', nearest_g.id, g_wp)] + [u.path[-1]]

    # global evaluation
    V_D_glob, V_U_glob, per_uav_lengths, per_uav_finish_times, feas = evaluate_all_uavs(uav_copies, iot_objs, gb_objs)
    aoi_dict, peak, avg, coverage = compute_aoi_from_VU(V_U_glob, iot_objs)

    # normalize terms
    peak_val = float('inf') if peak is None else float(peak)
    avg_val = float('inf') if avg is None else float(avg)
    # AoI normalization by Tmax
    peak_norm = (peak_val / float(T_MAX)) if peak_val != float('inf') else 1.0
    avg_norm = (avg_val / float(T_MAX)) if avg_val != float('inf') else 1.0

    total_dist = sum(per_uav_lengths.values()) if per_uav_lengths else 0.0
    max_dist_bound = len(uav_copies) * math.hypot(AREA_SIZE, AREA_SIZE)
    dist_norm = total_dist / max_dist_bound if max_dist_bound > 0 else 0.0

    # penalty: sum of overtime amounts across UAVs (finish_time - Tmax) normalized by Tmax
    overtime_sum = 0.0
    for uid, tfin in per_uav_finish_times.items():
        overtime_sum += max(0.0, tfin - T_MAX)
    overtime_norm = overtime_sum / (len(uav_copies) * float(T_MAX) + 1e-12)

    penalty_term = penalty_coeff * overtime_norm
    # Tmax penalty: add w_penalty for each UAV exceeding Tmax
    Tmax_penalty = sum(w_penalty for u_id, f in feas.items() if not f)
    # composite score (smaller is better)
    #score = (w_peak * peak_norm) + (w_avg * avg_norm) + (w_dist * dist_norm) + (w_pen * penalty_term)
    score = (w_peak * peak_norm) + (w_avg * avg_norm) + (w_dist * dist_norm) + Tmax_penalty

    return score, peak_val, avg_val, coverage, total_dist, per_uav_finish_times, feas, uav_copies

# ----------------------------
# Encoding: random-key + continuous breakpoints
# ----------------------------
def continuous_to_chrom_and_assign(X_cont: np.ndarray, num_iot: int, num_uav: int):
    keys = list(enumerate(X_cont[:num_iot]))
    keys_sorted = sorted(keys, key=lambda p: p[1])
    perm = [k[0] for k in keys_sorted]
    bps = []
    if num_uav > 1:
        bps_cont = X_cont[num_iot:num_iot + (num_uav - 1)]
        for v in bps_cont:
            s = math.tanh(float(v))
            scaled = int(round(((s + 1.0) / 2.0) * (num_iot - 2))) + 1 if num_iot > 2 else 1
            scaled = max(1, min(num_iot - 1, scaled))
            bps.append(scaled)
        bps = sorted(bps)
        for i in range(1, len(bps)):
            if bps[i] <= bps[i-1]:
                bps[i] = min(num_iot - 1, bps[i-1] + 1)
        while len(bps) < (num_uav - 1):
            candidate = min(num_iot - 1, (bps[-1] + 1) if bps else 1)
            bps.append(candidate)
        bps = bps[:(num_uav - 1)]
    # decode to assignment
    assign = [None] * num_iot
    start = 0
    for uidx in range(num_uav):
        if uidx < num_uav - 1:
            bp = bps[uidx]
            segment = perm[start:bp]
            start = bp
        else:
            segment = perm[start:]
        for i in segment:
            assign[i] = uidx
    for i in range(num_iot):
        if assign[i] is None:
            assign[i] = random.randrange(num_uav)
    return (perm, bps), assign

# ----------------------------
# Levy flight
# ----------------------------
def levy_flight(shape, beta=LEVY_BETA):
    num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2.0)
    den = math.gamma((1 + beta) / 2.0) * beta * 2 ** ((beta - 1.0) / 2.0)
    sigma_u = (num / den) ** (1.0 / beta)
    u = np.random.normal(0, sigma_u, size=shape)
    v = np.random.normal(0, 1, size=shape)
    step = u / (np.abs(v) ** (1.0 / beta))
    return step

# ----------------------------
# Initialization helper
# ----------------------------
def initialization(pop, dim, ub, lb):
    return np.random.uniform(lb, ub, size=(pop, dim))


def encode_chrom_to_continuous(chrom, num_iot, num_uav, lb=-2.0, ub=2.0):
    """
    chrom: (perm, bps)
    returns: X_cont numpy array of length num_iot + max(0, num_uav-1)
    Strategy:
      - For perm: assign increasing key values in [-0.9,0.9] according to perm order, then scale to [lb,ub]
      - For bps: map integer breakpoint positions -> continuous via inverse of the tanh scaling
    """
    perm, bps = chrom
    dim_perm = num_iot
    dim_bp = max(0, num_uav - 1)
    X = np.zeros(dim_perm + dim_bp, dtype=float)

    # Perm keys in order of positions
    if num_iot == 1:
        keys = [0.0]
    else:
        keys = np.linspace(-0.9, 0.9, num=num_iot)
    for pos, iot_idx in enumerate(perm):
        X[iot_idx] = keys[pos]

    # Breakpoints inverse mapping
    if dim_bp > 0:
        if num_iot <= 2:
            for j in range(dim_bp):
                X[dim_perm + j] = 0.0
        else:
            for j in range(dim_bp):
                if j < len(bps):
                    b = bps[j]
                    b = max(1, min(num_iot - 1, int(b)))
                    frac = (b - 1) / float(num_iot - 2)
                    target_s = frac * 2.0 - 1.0
                    target_s = max(-0.999999, min(0.999999, target_s))
                    v = 0.5 * math.log((1 + target_s) / (1 - target_s))  # arctanh
                    X[dim_perm + j] = float(v)
                else:
                    X[dim_perm + j] = 0.0

    # clip and scale to [lb,ub]
    X = np.clip(X, -1.5, 1.5)
    X_scaled = lb + (X + 1.5) * ((ub - lb) / 3.0)
    return X_scaled

def build_greedy_solution(iot_objs, gb_objs, uav_list, R_I=R_IOT, R_G=R_GBS):
    iot_map = {iot.id: iot for iot in iot_objs}
    unvisited = set(iot_map.keys())
    solution_uavs = {u.id: [('start', None, u.start)] for u in uav_list}
    for u in uav_list:
        cur_pos = u.start
        cur_time = 0
        while True:
            candidates = []
            for iid in list(unvisited):
                iobj = iot_map[iid]
                i_wp = point_on_line_at_distance_from_target(cur_pos, iobj.pos, R_I)
                t_arrive = cur_time + travel_time(cur_pos, i_wp, u.V)
                t_capture = max(t_arrive, iobj.t_gen)
                nearest_g = min(gb_objs, key=lambda g: dist(i_wp, g.pos))
                g_wp = point_on_line_at_distance_from_target(i_wp, nearest_g.pos, R_G)
                t_upload = t_capture + travel_time(i_wp, g_wp, u.V)
                if t_upload <= u.Tmax:
                    candidates.append((iid, i_wp, nearest_g.id, g_wp, t_upload))
            if not candidates:
                break
            candidates.sort(key=lambda x: dist(cur_pos, x[1]))
            iid, i_wp, gid, g_wp, t_upload = candidates[0]
            solution_uavs[u.id].append(('iot', iid, i_wp))
            if dist(i_wp, g_wp) < GB_IMMEDIATE_INSERT_THRESHOLD:
                solution_uavs[u.id].append(('gb', gid, g_wp))
                cur_pos = g_wp
            else:
                cur_pos = i_wp
            cur_time = int(t_upload)
            unvisited.discard(iid)
        solution_uavs[u.id].append(('end', None, u.end))
    # fallback assignment of remaining (rare)
    if unvisited:
        for iid in list(unvisited):
            iobj = iot_map[iid]
            best_u = min(uav_list, key=lambda uu: dist(solution_uavs[uu.id][-2][2], iobj.pos))
            prev_pos = solution_uavs[best_u.id][-2][2]
            i_wp = point_on_line_at_distance_from_target(prev_pos, iobj.pos, R_I)
            nearest_g = min(gb_objs, key=lambda g: dist(i_wp, g.pos))
            g_wp = point_on_line_at_distance_from_target(i_wp, nearest_g.pos, R_G)
            solution_uavs[best_u.id] = solution_uavs[best_u.id][:-1] + [('iot', iid, i_wp), ('gb', nearest_g.id, g_wp), ('end', None, best_u.end)]
            unvisited.discard(iid)
    for u in uav_list:
        solution_uavs[u.id] = clean_path_remove_consecutive_gbs(solution_uavs[u.id], u)
    return solution_uavs

# ----------------------------
# SFW main loop (adapted)
# ----------------------------
def run_sfw(iot_objs, gb_objs, uav_list, pop=POP, max_iters=MAX_ITERS):
    num_iot = len(iot_objs)
    num_uav = len(uav_list)
    dim = num_iot + max(0, num_uav - 1)
    lb = -2.0
    ub = 2.0
    X = initialization(pop, dim, ub, lb)
    fitness = np.full(pop, float('inf'))
    best_fitness = float('inf')
    best_position = None
    best_curve = []
    eval_cache = {}

    # Optional greedy seed (encode greedy path into continuous keys)
    SEED_WITH_GREEDY = True
    if SEED_WITH_GREEDY:
        greedy_sol = build_greedy_solution(iot_objs, gb_objs, uav_list)
        # extract (perm,bps) from greedy_sol
        perm_list = []
        bps_list = []
        for u in uav_list:
            seg = []
            for typ, idx, pos in greedy_sol[u.id]:
                if typ == 'iot':
                    seg.append(idx - 1)  # IoT id -> zero-based index
            perm_list.extend(seg)
            if u.id != uav_list[-1].id:
                bps_list.append(len(perm_list))
        # append any missing IoTs
        all_idx = set(range(num_iot))
        missing = list(all_idx - set(perm_list))
        if missing:
            perm_list.extend(missing)
        # trim/pad breakpoints
        while len(bps_list) < max(0, num_uav - 1):
            last = bps_list[-1] if bps_list else 0
            candidate = min(num_iot - 1, last + 1)
            bps_list.append(candidate)
        bps_list = bps_list[:max(0, num_uav - 1)]
        greedy_chrom = (perm_list, bps_list)
        X_greedy = encode_chrom_to_continuous(greedy_chrom, num_iot, num_uav, lb=lb, ub=ub)
        for i in range(round(pop / 2)):
            X[i] = X_greedy
        n_seed_perturb = min(3, pop - 1)
        for k in range(1, 1 + n_seed_perturb):
            noise = np.random.normal(scale=0.05, size=dim)
            X[k] = np.clip(X_greedy + noise, lb, ub)

    def eval_cont_vector(x_cont):
        chrom, assign = continuous_to_chrom_and_assign(x_cont, num_iot, num_uav)
        key = (tuple(chrom[0]), tuple(chrom[1]))
        if key in eval_cache:
            return eval_cache[key]
        res = evaluate_assignment_fitness(assign, iot_objs, gb_objs, uav_list)
        eval_cache[key] = res + (chrom, assign)
        return eval_cache[key]

    # initial eval
    for i in range(pop):
        score, *_ = eval_cont_vector(X[i])
        fitness[i] = score
        if score < best_fitness:
            best_fitness = score
            best_position = X[i].copy()
    best_curve.append(best_fitness)

    L = levy_flight((pop, dim), beta=LEVY_BETA)
    for it in range(1, max_iters + 1):
        w = (math.pi / 2.0) * (it / (max_iters + 1e-12))
        r1 = random.random()
        r2 = random.random()
        k = 0.2 * math.sin(math.pi / 2.0 - w)
        m = it / float(max_iters) * 2.0
        LB = np.ones(dim) * lb
        UB = np.ones(dim) * ub
        p = np.sin(UB - LB) * 2.0 + (UB - LB) * m
        Xnew = X.copy()
        for i in range(pop):
            T = 0.5
            c1 = random.random()
            if T < c1:
                Xnew[i] = X[i] + (LB + (UB - LB) * np.random.rand(dim))
            else:
                s = r1 * 20.0 + r2 * 20.0
                if s > 20.0:
                    y = random.randrange(pop)
                    if best_position is None:
                        continue
                    Xnew[i] = best_position + X[i] * L[y] * k
                else:
                    if best_position is None:
                        continue
                    XG = best_position * C_PARAM
                    Xnew[i] = XG + (best_position - X[i]) * p
            Xnew[i] = np.maximum(Xnew[i], lb)
            Xnew[i] = np.minimum(Xnew[i], ub)
        # evaluate Xnew and replace if better
        for i in range(pop):
            score, *_ = eval_cont_vector(Xnew[i])
            if score < fitness[i]:
                X[i] = Xnew[i]
                fitness[i] = score
                if score < best_fitness:
                    best_fitness = score
                    best_position = X[i].copy()



        # ----------------------------
        # AFGMS: Adaptive Fitness-Guided Mutation Strategy
        # ----------------------------
        # compute population stats
        fmin = float(np.min(fitness))
        fmax = float(np.max(fitness))
        fmean = float(np.mean(fitness))
        # decay factor for mutation magnitude (stronger earlier)
        decay = (1.0 - (it / float(max_iters)))  # linear decay [1..0]
        adapt_scale = SCALE0 * decay

        for i in range(pop):
            fi = fitness[i]
            # compute adaptive mutation probability pm
            if fi <= fmean:
                # normalized ratio: 0 (best) -> 1 (close to mean)
                denom = (fmean - fmin) + EPS
                ratio = max(0.0, min(1.0, (fi - fmin) / denom))
                pm = PM_LOW + (PM_HIGH - PM_LOW) * (ratio ** GAMMA)
            else:
                pm = PM_HIGH
            # perform mutation with probability pm
            if random.random() < pm:
                # hybrid mutation: Gaussian + scaled Cauchy (heavy-tailed)
                gaussian = np.random.randn(dim) * SIGMA_G
                # Cauchy via standard_cauchy
                cauchy = np.random.standard_cauchy(size=dim) * SIGMA_C
                noise = adapt_scale * (gaussian + cauchy)
                Xmut = X[i] + noise
                Xmut = np.maximum(Xmut, lb)
                Xmut = np.minimum(Xmut, ub)
                # evaluate and accept if better (greedy acceptance)
                score_mut, *_ = eval_cont_vector(Xmut)
                if score_mut < fitness[i]:
                    X[i] = Xmut
                    fitness[i] = score_mut
                    if score_mut < best_fitness:
                        best_fitness = score_mut
                        best_position = X[i].copy()



        best_curve.append(best_fitness)
        if it % 10 == 0 or it == max_iters:
            info = eval_cont_vector(best_position)
            print(f"Iter {it}/{max_iters}: best_score={best_fitness:.6f}, peak={info[1]}, avg={info[2]}, coverage={info[3]:.3f}")
    best_res = eval_cont_vector(best_position)
    best_chrom = best_res[8]
    best_assign = best_res[9]
    return best_chrom, best_assign, best_res, best_curve

# ----------------------------
# Scenario build + plot
# ----------------------------
def build_random_scenario(num_iot=NUM_IOT, num_gbs=NUM_GBS, num_uavs=NUM_UAV, area=AREA_SIZE, seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    gb_list = []
    for gid in range(1, num_gbs + 1):
        x = random.uniform(0.1 * area, 0.9 * area)
        y = random.uniform(0.1 * area, 0.9 * area)
        gb_list.append(GBS(gid, x, y))
    iot_list = []
    for iid in range(1, num_iot + 1):
        x = random.uniform(0, area)
        y = random.uniform(0, area)
        tgen = random.randint(0, 20)
        iot_list.append(IoT(iid, x, y, tgen))
    starts = [(0.05 * area, 0.05 * area), (0.95 * area, 0.05 * area), (0.05 * area, 0.95 * area), (0.95 * area, 0.95 * area)]
    ends = [(0.05 * area, 0.95 * area), (0.95 * area, 0.95 * area), (0.95 * area, 0.05 * area), (0.05 * area, 0.05 * area)]
    uav_list = []
    for uid in range(1, num_uavs + 1):
        s = starts[(uid - 1) % len(starts)]
        e = ends[(uid - 1) % len(ends)]
        uav_list.append(UAV(uid, s, e, UAV_SPEED, T_MAX))
    return iot_list, gb_list, uav_list

def plot_solution(iot_list, gb_list, uav_list, title="SFW solution"):
    plt.figure(figsize=(8, 8))
    ixs = [iot.pos[0] for iot in iot_list]
    iys = [iot.pos[1] for iot in iot_list]
    plt.scatter(ixs, iys, marker='o', label='IoT')
    for iot in iot_list:
        plt.text(iot.pos[0] + 0.4, iot.pos[1] + 0.4, f"I{iot.id}(t={iot.t_gen})", fontsize=8)
    gxs = [g.pos[0] for g in gb_list]
    gys = [g.pos[1] for g in gb_list]
    plt.scatter(gxs, gys, marker='s', label='GBS')
    for g in gb_list:
        plt.text(g.pos[0] + 0.4, g.pos[1] + 0.4, f"G{g.id}", fontsize=9)
    colors = ['green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    for u in uav_list:
        path_coords = [np.array(w[2]) for w in u.path]
        px = [p[0] for p in path_coords]
        py = [p[1] for p in path_coords]
        plt.plot(px, py, '--x', label=f"UAV{u.id}", color=colors[(u.id - 1) % len(colors)])
        plt.text(u.start[0], u.start[1], f"U{u.id}_S", fontsize=9)
        plt.text(u.end[0], u.end[1], f"U{u.id}_F", fontsize=9)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlim(0, AREA_SIZE)
    plt.ylim(0, AREA_SIZE)
    plt.show()

# ----------------------------
# Run demo
# ----------------------------
if __name__ == '__main__':
    iot_objs, gb_list, uav_list = build_random_scenario(num_iot=NUM_IOT, num_gbs=NUM_GBS, num_uavs=NUM_UAV, seed=SEED)
    print(f"Scenario: IoTs={len(iot_objs)}, GBSs={len(gb_list)}, UAVs={len(uav_list)}")
    t0 = time.time()
    best_chrom, best_assign, best_res, curve = run_sfw(iot_objs, gb_list, uav_list, pop=POP, max_iters=MAX_ITERS)
    elapsed = time.time() - t0
    print(f"SFW done in {elapsed:.2f}s: best_score={best_res[0]:.6f}")
    print("Best info (score, peak, avg, coverage, tot_dist):", best_res[0], best_res[1], best_res[2], best_res[3], best_res[4])
    uav_copies = best_res[7]
    V_D_glob, V_U_glob, per_uav_lengths, per_uav_finish_times, feas_flags = evaluate_all_uavs(uav_copies, iot_objs, gb_list)
    aoi_dict, peak, avg, coverage = compute_aoi_from_VU(V_U_glob, iot_objs)
    print("Per-UAV finish times:", per_uav_finish_times)
    print("Per-UAV feasible:", feas_flags)
    print(f"Peak AoI: {peak}, Avg AoI: {avg}, Coverage: {coverage:.3f}")
    for u in uav_copies:
        print(f"UAV{u.id} path:")
        for typ, idx, pos in u.path:
            print(f"  {typ:5s} {str(idx):>4s} @ ({pos[0]:.2f},{pos[1]:.2f})")
        print("")
    plot_solution(iot_objs, gb_list, uav_copies, title="SFW multi-UAV (weighted fitness) result")
