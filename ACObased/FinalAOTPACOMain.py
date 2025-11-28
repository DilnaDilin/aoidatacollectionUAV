#!/usr/bin/env python3
"""
ACO for multi-UAV AoI-aware IoT collection (range-aware) - cleaned & fixed

Key points:
 - Fitness depends ONLY on Peak AoI and Average AoI (normalized).
 - No 2-opt / local search (per your request).
 - Ants build feasible tours visiting reachable IoTs; remaining IoTs assigned to
   nearest UAV at the end (fallback).
 - Pheromone matrix on directed IoT edges + start->node tau.
 - All ants deposit pheromone each iteration (deposit = 1/(fitness+eps)).
 - Optional elitist (best-so-far) reinforcement included.
 - Ensures each UAV path ends with ('end', None, u.end).
 - Robust probability selection, no division-by-zero, and early stopping logic correct.
"""
import random
import copy
import math
import time
from typing import List, Dict, Tuple
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

# ----------------------------
# ACO hyperparameters / tuning
# ----------------------------
NUM_ANTS = 30
ACO_ITERS = 50
ALPHA = 1.0     # pheromone importance
BETA = 2.0      # heuristic importance
RHO = 0.12      # evaporation rate
LAMBDA = 1.0    # ant deposit scalar
ELITIST_BOOST = 6.0  # multiplier for best-so-far reinforcement (set 0 to disable)
NO_IMPROVE_LIMIT = 120
TAU0 = 1.0      # initial pheromone
SEED_WITH_GREEDY = True  # if True, seed pheromone from greedy
GB_IMMEDIATE_INSERT_THRESHOLD = 2 * R_GBS  # threshold to insert GB immediately

# Fitness weights (normalized)
W_PEAK = 0.7
W_AVG = 0.3
EPS = 1e-9  # numerical stability

# ----------------------------
# Utilities
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
    norm = np.linalg.norm(vec)
    if np.isclose(norm, 0.0):
        return tuple(tgt + np.array([radius, 0.0]))
    unit = vec / norm
    return tuple(tgt + unit * radius)

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
# Evaluation functions
# ----------------------------
def evaluate_uav_path(uav, iot_objs, gb_objs, R_I=R_IOT, R_G=R_GBS):
    """
    Simulate UAV path. Returns:
      V_D: dict IoT id -> capture time (or None)
      V_U: dict IoT id -> upload time (or None)
      total_dist, finish_time, feasible (<=Tmax)
    """
    V_D = {iot.id: None for iot in iot_objs}
    V_U = {iot.id: None for iot in iot_objs}
    iot_tgen = {iot.id: iot.t_gen for iot in iot_objs}
    cur_pos = np.array(uav.path[0][2], dtype=float)  # start
    t = 0
    total_dist = 0.0
    captured_pending = []

    # iterate waypoints (skip initial 'start' since cur_pos already there)
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

    # if pending, try to upload by flying to nearest GBS
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
        # merge (first writer wins)
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
# Greedy builder (optional seeding)
# ----------------------------

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
# Pheromone initialization
# ----------------------------
def init_pheromone(n, tau0=TAU0):
    pher = np.full((n, n), float(tau0))
    np.fill_diagonal(pher, 0.0)
    start_tau = np.full(n, float(tau0))
    return pher, start_tau

# ----------------------------
# Heuristic: inverse cost combining distance and expected upload time
# ----------------------------
def heuristic_value(cur_pos, candidate_wp, t_now, iobj: IoT, gb_objs, uav_v):
    d = dist(cur_pos, candidate_wp)
    t_arrive = t_now + travel_time(cur_pos, candidate_wp, uav_v)
    t_capture = max(t_arrive, iobj.t_gen)
    nearest_g = min(gb_objs, key=lambda g: dist(candidate_wp, g.pos))
    g_wp = point_on_line_at_distance_from_target(candidate_wp, nearest_g.pos, R_GBS)
    t_upload = t_capture + travel_time(candidate_wp, g_wp, uav_v)
    cost = d + 0.8 * (t_capture - t_now) + 0.6 * (t_upload - t_now)
    return 1.0 / (cost + 1.0)

# ----------------------------
# Construct solution for one ant
# ----------------------------
def construct_solution(pheromone, start_tau, iot_list, gb_list, uav_list, R_I=R_IOT, R_G=R_GBS):
    n = len(iot_list)
    iot_map = {iot_list[i].id: iot_list[i] for i in range(n)}
    all_unvisited = set(iot_map.keys())
    solution_uavs = {u.id: [('start', None, u.start)] for u in uav_list}

    for u in uav_list:
        cur_pos = u.start
        cur_time = 0
        prev_idx = None  # previous IoT index (0-based)
        chosen_order = []
        while True:
            candidates = []
            for iid in list(all_unvisited):
                iobj = iot_map[iid]
                i_wp = point_on_line_at_distance_from_target(cur_pos, iobj.pos, R_I)
                t_arrive = cur_time + travel_time(cur_pos, i_wp, u.V)
                t_capture = max(t_arrive, iobj.t_gen)
                nearest_g = min(gb_list, key=lambda g: dist(i_wp, g.pos))
                g_wp = point_on_line_at_distance_from_target(i_wp, nearest_g.pos, R_G)
                t_upload = t_capture + travel_time(i_wp, g_wp, u.V)
                if t_upload <= u.Tmax:
                    candidates.append((iid, i_wp, nearest_g.id, g_wp, t_upload))
            if not candidates:
                break

            probs = []
            denom = 0.0
            for (iid, i_wp, gid, g_wp, t_upload) in candidates:
                cur_idx = iid - 1
                tau = start_tau[cur_idx] if prev_idx is None else pheromone[prev_idx, cur_idx]
                heur = heuristic_value(cur_pos, i_wp, cur_time, iot_map[iid], gb_list, u.V)
                val = (tau ** ALPHA) * (heur ** BETA)
                probs.append(val)
                denom += val

            if denom <= 0:
                break
            # roulette wheel selection
            probs = [p / denom for p in probs]
            sel = np.random.choice(len(candidates), p=probs)
            iid, i_wp, gid, g_wp, t_upload = candidates[int(sel)]

            solution_uavs[u.id].append(('iot', iid, i_wp))
            chosen_order.append(iid)

            if dist(i_wp, g_wp) < GB_IMMEDIATE_INSERT_THRESHOLD:
                solution_uavs[u.id].append(('gb', gid, g_wp))
                cur_pos = g_wp
            else:
                cur_pos = i_wp

            cur_time = int(t_upload)
            prev_idx = iid - 1
            all_unvisited.discard(iid)

        # ensure end and cleaning
        solution_uavs[u.id].append(('end', None, u.end))
        solution_uavs[u.id] = clean_path_remove_consecutive_gbs(solution_uavs[u.id], u)

    # fallback: assign any remaining unvisited to nearest UAVs
    if all_unvisited:
        for iid in list(all_unvisited):
            iobj = iot_map[iid]
            best_u = min(uav_list, key=lambda uu: dist(solution_uavs[uu.id][-2][2], iobj.pos))
            prev_pos = solution_uavs[best_u.id][-2][2]
            i_wp = point_on_line_at_distance_from_target(prev_pos, iobj.pos, R_I)
            nearest_g = min(gb_list, key=lambda g: dist(i_wp, g.pos))
            g_wp = point_on_line_at_distance_from_target(i_wp, nearest_g.pos, R_G)
            solution_uavs[best_u.id] = solution_uavs[best_u.id][:-1] + [('iot', iid, i_wp), ('gb', nearest_g.id, g_wp), ('end', None, best_u.end)]
            solution_uavs[best_u.id] = clean_path_remove_consecutive_gbs(solution_uavs[best_u.id], best_u)
            all_unvisited.discard(iid)

    return solution_uavs

# ----------------------------
# Fitness: depends ONLY on peak and avg AoI (normalized). Lower is better.
# ----------------------------

def compute_fitness_for_solution(solution_uavs, iot_objs, gb_objs, base_uavs, w_peak=W_PEAK, w_avg=W_AVG, w_penalty=1000.0):

    #Computes fitness for a set of UAV paths. Lower is better.#
    #Adds a penalty if any UAV exceeds Tmax.
    #Returns fitness, peak, avg, coverage, total distance, per_uav_finish_times, feasibility flags, and UAV copies.

    # build uav copies and assign paths
    uav_copies = [u.copy() for u in base_uavs]
    for u in uav_copies:
        u.path = solution_uavs[u.id]

    # finalize pending uploads by adding nearest GBS before end if needed
    for u in uav_copies:
        V_D, V_U, plen, tfin, feas = evaluate_uav_path(u, iot_objs, gb_objs)
        pending = [iid for iid in V_D if V_D[iid] is not None and V_U[iid] is None]
        if pending:
            nearest_g = min(gb_objs, key=lambda g: dist(u.path[-2][2], g.pos))
            g_wp = point_on_line_at_distance_from_target(u.path[-2][2], nearest_g.pos, R_GBS)
            u.path = u.path[:-1] + [('gb', nearest_g.id, g_wp)] + [u.path[-1]]

    # evaluate globally
    V_D_glob, V_U_glob, per_uav_lengths, per_uav_finish_times, feas = evaluate_all_uavs(uav_copies, iot_objs, gb_objs)
    aoi_dict, peak, avg, coverage = compute_aoi_from_VU(V_U_glob, iot_objs)

    # Normalize peak and avg by Tmax
    peak_val = float('inf') if peak is None else float(peak)
    avg_val = float('inf') if avg is None else float(avg)

    peak_norm = (peak_val / float(T_MAX)) if peak_val != float('inf') else 1.0
    avg_norm = (avg_val / float(T_MAX)) if avg_val != float('inf') else 1.0

    # Tmax penalty: add w_penalty for each UAV exceeding Tmax
    Tmax_penalty = sum(w_penalty for u_id, f in feas.items() if not f)

    # total fitness
    fitness = w_peak * peak_norm + w_avg * avg_norm + Tmax_penalty

    return fitness, peak, avg, coverage, sum(per_uav_lengths.values()), per_uav_finish_times, feas, uav_copies


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
# ACO main loop
# ----------------------------
def run_aco(iot_objs, gb_objs, uav_list):
    random.seed(SEED)
    np.random.seed(SEED)
    n = len(iot_objs)
    pheromone, start_tau = init_pheromone(n, TAU0)

    # optional greedy seed
    if SEED_WITH_GREEDY:
        greedy_sol = build_greedy_solution(iot_objs, gb_objs, uav_list)
        # reinforce edges from greedy
        boost = TAU0 * 5.0
        for u in uav_list:
            prev_idx = None
            for typ, idx, pos in greedy_sol[u.id]:
                if typ == 'iot':
                    cur_idx = int(idx) - 1
                    if prev_idx is None:
                        start_tau[cur_idx] += boost
                    else:
                        pheromone[prev_idx, cur_idx] += boost
                    prev_idx = cur_idx
                elif typ == 'start':
                    prev_idx = None

    best_solution = None
    best_fitness = float('inf')
    best_info = None
    history = []
    no_improve = 0

    for it in range(ACO_ITERS):
        prev_best = best_fitness
        ants_solutions = []
        ants_fitnesses = []

        # each ant constructs a solution and we evaluate fitness
        for a in range(NUM_ANTS):
            sol = construct_solution(pheromone, start_tau, iot_objs, gb_objs, uav_list)
            fitness, peak, avg, coverage, total_len, per_uav_finish_times, feas, uav_copies = compute_fitness_for_solution(sol, iot_objs, gb_objs, uav_list)
            ants_solutions.append((sol, fitness, peak, avg, coverage, total_len, per_uav_finish_times, feas, uav_copies))
            ants_fitnesses.append(fitness)
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = sol
                best_info = (fitness, peak, avg, coverage, total_len, per_uav_finish_times, feas, uav_copies)

        # check improvement vs previous iter
        if best_fitness >= prev_best - 1e-12:
            no_improve += 1
        else:
            no_improve = 0

        # evaporation
        pheromone *= (1.0 - RHO)
        start_tau *= (1.0 - RHO)

        # deposit from each ant (proportional to 1/(fitness+eps))
        delta = np.zeros_like(pheromone)
        delta_start = np.zeros_like(start_tau)
        for sol, fitness, *_ in ants_solutions:
            deposit = LAMBDA / (fitness + EPS)
            # walk through each UAV path and deposit on directed edges
            for u in uav_list:
                prev_idx = None
                for typ, idx, pos in sol[u.id]:
                    if typ == 'iot':
                        cur_idx = int(idx) - 1
                        if prev_idx is None:
                            delta_start[cur_idx] += deposit
                        else:
                            delta[prev_idx, cur_idx] += deposit
                        prev_idx = cur_idx
                    elif typ == 'start':
                        prev_idx = None

       # # elitist (best-so-far) deposit
       # if best_solution is not None and ELITIST_BOOST > 0:
       #     best_deposit = ELITIST_BOOST * (1.0 / (best_fitness + EPS))
       #     for u in uav_list:
       #         prev_idx = None
       #         for typ, idx, pos in best_solution[u.id]:
       #             if typ == 'iot':
       #                 cur_idx = int(idx) - 1
       #                 if prev_idx is None:
       #                     delta_start[cur_idx] += best_deposit
       #                 else:
       #                     delta[prev_idx, cur_idx] += best_deposit
       #                 prev_idx = cur_idx
       #             elif typ == 'start':
       #                 prev_idx = None

        # apply deposits
        pheromone += delta
        start_tau += delta_start
        # numerical bounds
        np.clip(pheromone, 1e-9, 1e9, out=pheromone)
        np.clip(start_tau, 1e-9, 1e9, out=start_tau)

        if it % 5 == 0 or it == ACO_ITERS - 1:
            if best_info is None:
                print(f"Iter {it}: no solution yet")
            else:
                bf = best_info
                print(f"Iter {it}: best_fitness={best_fitness:.6f}, peak={bf[1]}, avg={bf[2]}, total_dist={bf[4]:.2f}, coverage={bf[3]:.3f}")

        history.append((it, best_fitness, best_info))

        if no_improve >= NO_IMPROVE_LIMIT:
            print(f"No improvement for {NO_IMPROVE_LIMIT} iters -> early stop at iter {it}")
            break

    return best_solution, best_fitness, best_info, history

# ----------------------------
# Scenario builder & plotting (same as before)
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

def plot_solution(iot_list, gb_list, uav_list, title="ACO improved"):
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
# Main driver





import datetime
import csv
import time

# ----------------------------
# Multi-seed experiment driver
# ----------------------------
def run_multi_seed_experiment_aco(seeds_list, num_iot=NUM_IOT, num_gbs=NUM_GBS, num_uavs=NUM_UAV):
    results = []

    for s in seeds_list:
        print("\n==============================")
        print(f" Running Experiment - Seed {s}")
        print("==============================")

        # Build scenario
        iot_objs, gb_objs, uav_list = build_random_scenario(num_iot=num_iot, num_gbs=num_gbs, num_uavs=num_uavs, seed=s)
        print(f"Scenario: IoTs={len(iot_objs)}, GBSs={len(gb_objs)}, UAVs={len(uav_list)}")

        # Run ACO
        t0 = time.time()
        best_sol, best_fitness, best_info, history = run_aco(iot_objs, gb_objs, uav_list)
        elapsed = time.time() - t0
        print(f"ACO done in {elapsed:.2f}s: best_fitness={best_fitness:.6f}")

        if best_info is not None:
            bf = best_info
            print(f"Best info (fitness, peak, avg, coverage, total_dist): {bf[0]:.6f}, {bf[1]}, {bf[2]}, {bf[3]:.3f}, {bf[4]:.1f}")

        if best_sol is not None:
            # Detailed evaluation
            fitness, peak, avg, coverage, total_dist, per_uav_finish_times, feas_flags, uav_copies = compute_fitness_for_solution(
                best_sol, iot_objs, gb_objs, uav_list
            )
            V_D_glob, V_U_glob, per_uav_lengths, per_uav_finish_times, feas_flags = evaluate_all_uavs(uav_copies, iot_objs, gb_objs)
            aoi_dict, peak, avg, coverage = compute_aoi_from_VU(V_U_glob, iot_objs)

            print("Per-UAV finish times:", per_uav_finish_times)
            print("Per-UAV feasible:", feas_flags)
            print(f"Peak AoI: {peak}, Avg AoI: {avg}, Coverage: {coverage:.3f}")

            # AoI per IoT
            for iid in sorted(aoi_dict.keys()):
                tgen = next(i.t_gen for i in iot_objs if i.id == iid)
                print(f" IoT {iid}: AoI = {aoi_dict[iid]}  (t_gen={tgen})")

            # UAV path lengths
            print("\n===== UAV Path Lengths =====")
            total_uav_path_len = 0.0
            for i, length in per_uav_lengths.items():
                print(f"UAV {i} path length: {length:.3f}")
                total_uav_path_len += length
            print(f"Total UAV path length: {total_uav_path_len:.3f}")

            # UAV paths
            for u in uav_copies:
                print(f"\nUAV{u.id} path:")
                for typ, idx, pos in u.path:
                    print(f"  {typ:5s} {str(idx):>4s} @ ({pos[0]:.2f},{pos[1]:.2f})")

            # Unserved IoTs
            served_iots = {iot_id for iot_id, v in V_U_glob.items() if v is not None}
            all_ids = set(iot.id for iot in iot_objs)
            not_served_ids = all_ids - served_iots
            print(f"\nNot served IoTs: {not_served_ids}, total: {len(not_served_ids)}")

            # Plot solution
           # plot_solution(iot_objs, gb_objs, uav_copies, title=f"ACO multi-UAV result | Seed {s}")

            # Save results
            results.append({
                "seed": s,
                "fitness": fitness,
                "peak_aoi": peak,
                "avg_aoi": avg,
                "coverage": coverage,
                "total_dist": total_dist,
                "unserved": len(not_served_ids),
                "runtime_sec": elapsed,
                "aoi_dict": aoi_dict,
                "t_gen_map": {i.id: i.t_gen for i in iot_objs},
                "per_uav_finish_times": per_uav_finish_times,
                "feas_flags": feas_flags,
                "uav_paths": {u.id: u.path for u in uav_copies}
            })
        else:
            print("No feasible solution produced by ACO.")
            results.append({"seed": s, "fitness": None, "peak_aoi": None, "avg_aoi": None,
                            "coverage": None, "total_dist": None, "unserved": NUM_IOT,
                            "runtime_sec": elapsed, "aoi_dict": {}, "t_gen_map": {},
                            "per_uav_finish_times": {}, "feas_flags": {}, "uav_paths": {}})

    return results

# ----------------------------
# Compute aggregated stats
# ----------------------------
def compute_final_stats(results):
    def avg(field):
        vals = [r[field] for r in results if r[field] is not None]
        return sum(vals) / len(vals) if vals else None

    final = {
        "avg_fitness": avg("fitness"),
        "avg_peak_aoi": avg("peak_aoi"),
        "avg_avg_aoi": avg("avg_aoi"),
        "avg_coverage": avg("coverage"),
        "avg_total_dist": avg("total_dist"),
        "avg_unserved": avg("unserved"),
        "avg_runtime": avg("runtime_sec"),
    }
    return final

# ----------------------------
# Save results to CSV
# ----------------------------
def save_results_csv(results, final_stats, num_iot=NUM_IOT, num_uavs=NUM_UAV, num_gbs=NUM_GBS):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"aco_multi_seed_results_{timestamp}.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # Scenario info
        writer.writerow(["Scenario Information"])
        writer.writerow(["Number of IoT Nodes", num_iot])
        writer.writerow(["Number of UAVs", num_uavs])
        writer.writerow(["Number of Ground BS", num_gbs])
        writer.writerow([])

        # Header
        writer.writerow(["Seed", "Fitness", "Peak AOI", "Avg AOI", "Coverage",
                         "Total Distance", "Unserved IoT", "Runtime (s)"])

        # Seed results
        for r in results:
            writer.writerow([r["seed"], r["fitness"], r["peak_aoi"], r["avg_aoi"], r["coverage"],
                             r["total_dist"], r["unserved"], r["runtime_sec"]])
            # IoT-level AoI
            writer.writerow(["Individual AoIs for Seed", r["seed"]])
            for iid in sorted(r["aoi_dict"].keys()):
                writer.writerow([f"IoT {iid}", r["aoi_dict"][iid], f"t_gen={r['t_gen_map'][iid]}"])
            writer.writerow([])

        # Final aggregated metrics
        writer.writerow(["Final Averages"])
        for k, v in final_stats.items():
            writer.writerow([k, v])

    print(f"\n📁 Detailed results saved to: {filename}")

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    seeds = [42, 50, 55, 60, 65, 85, 90, 105, 49, 110]  # modify as needed

    results = run_multi_seed_experiment_aco(seeds)
    final_stats = compute_final_stats(results)

    print("\n===== Aggregated Stats =====")
    for k, v in final_stats.items():
        print(f"{k}: {v}")

    save_results_csv(results, final_stats)
