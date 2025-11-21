# aco_multi_uav_im_with_gbs_improved.py
"""
Improved ACO for multi-UAV AoI-aware IoT collection (range-aware).

Key fixes / improvements:
 - Fix cur_pos / cur_time advancement (no teleporting).
 - Use edge pheromone pheromone[i,j] and start_tau for start->node.
 - Seed pheromone from greedy deterministic solution (optional).
 - AoI-aware heuristic: considers distance, t_gen and estimated upload time.
 - Pheromone deposit rewards AoI directly (better avg AoI -> larger deposit).
 - Elitist deposit (best-so-far) to stabilize learning.
 - Per-UAV 2-opt local search on IoT order to refine tours.
 - Insert GBS immediately only if close enough (configurable).
 - Early stopping when no improvement.

Usage:
    python aco_multi_uav_im_with_gbs_improved.py

Requires: numpy, matplotlib (for plotting if desired)
"""

import random
import copy
import math
import time
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Problem parameters (match your GA/greedy runs)
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
ACO_ITERS = 120
ALPHA = 1.0     # pheromone importance
BETA = 2.0      # heuristic importance
RHO = 0.12      # evaporation
LAMBDA = 1.0    # ant deposit scalar
ELITIST_BOOST = 6.0  # multiplier for best-so-far reinforcement
NO_IMPROVE_LIMIT = 120
TAU0 = 1.0      # initial pheromone
SEED_WITH_GREEDY = True  # whether to seed pheromone with greedy solution
GB_IMMEDIATE_INSERT_THRESHOLD = 2 * R_GBS  # insert GB immediately only if dist < threshold

# ----------------------------
# Utilities (same as before)
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
# Evaluator (same semantics as earlier)
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
                t += int(iot_tgen[idx] - t)
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
# Small TSP 2-opt for improving IoT order of a UAV
# ----------------------------
def two_opt_order(order: List[int], pos_map: Dict[int, Tuple[float,float]]):
    if len(order) <= 2:
        return order
    best = order[:]
    def length(ordr):
        L = 0.0
        for a,b in zip(ordr, ordr[1:]):
            L += dist(pos_map[a], pos_map[b])
        return L
    improved = True
    # limit iterations for speed
    iters = 0
    while improved and iters < 30:
        improved = False
        iters += 1
        n = len(best)
        for i in range(0, n-2):
            for j in range(i+2, n):
                new = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                if length(new) + 1e-9 < length(best):
                    best = new
                    improved = True
        # small speed safeguard
    return best

# ----------------------------
# Small greedy builder (deterministic) used for seeding pheromone
#    For seeding we use a simple feasible nearest-first per UAV builder.
# ----------------------------
def build_greedy_solution(iot_objs, gb_objs, uav_list, R_I=R_IOT, R_G=R_GBS):
    iot_map = {iot.id: iot for iot in iot_objs}
    unvisited = set(iot_map.keys())
    solution_uavs = {u.id: [('start', None, u.start)] for u in uav_list}
    for u in uav_list:
        cur_pos = u.start
        cur_time = 0
        while True:
            # find nearest feasible IoT
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
            # pick nearest by distance from cur_pos
            candidates.sort(key=lambda x: dist(cur_pos, x[1]))
            iid, i_wp, gid, g_wp, t_upload = candidates[0]
            solution_uavs[u.id].append(('iot', iid, i_wp))
            if dist(i_wp, g_wp) < GB_IMMEDIATE_INSERT_THRESHOLD:
                solution_uavs[u.id].append(('gb', gid, g_wp))
                cur_pos = g_wp
            else:
                cur_pos = i_wp
            cur_time = t_upload
            unvisited.discard(iid)
        solution_uavs[u.id].append(('end', None, u.end))
        solution_uavs[u.id] = clean_path_remove_consecutive_gbs(solution_uavs[u.id], u)
    # fallback: assign remaining to nearest UAVs (should be none usually)
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
    return solution_uavs

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
# Initialize pheromone: edge matrix (n x n) and start_tau vector (n)
# ----------------------------
def init_pheromone(n, tau0=TAU0):
    pher = np.full((n, n), float(tau0))
    np.fill_diagonal(pher, 0.0)
    start_tau = np.full(n, float(tau0))
    return pher, start_tau

# ----------------------------
# AoI-aware heuristic: considers distance, waiting (t_gen), and estimated upload time
# ----------------------------
def heuristic_value(cur_pos, candidate_wp, t_now, iobj: IoT, gb_objs, uav_v):
    # distance cost
    d = dist(cur_pos, candidate_wp)
    # estimate arrival
    t_arrive = t_now + travel_time(cur_pos, candidate_wp, uav_v)
    t_capture = max(t_arrive, iobj.t_gen)
    # estimate nearest GBS upload time
    nearest_g = min(gb_objs, key=lambda g: dist(candidate_wp, g.pos))
    g_wp = point_on_line_at_distance_from_target(candidate_wp, nearest_g.pos, R_GBS)
    t_upload = t_capture + travel_time(candidate_wp, g_wp, uav_v)
    # we want small (d + waiting + upload_time). Construct heuristic as inverse of a combined cost
    wait = max(0, iobj.t_gen - t_now)
    cost = d + 0.8 * (t_capture - t_now) + 0.6 * (t_upload - t_now)
    # avoid division by zero
    return 1.0 / (cost + 1.0)

# ----------------------------
# Build solution for one ant (construct_solution) using edge pheromone and start_tau.
# Important: Fix cur_pos advancement correctly (no teleporting).
# ----------------------------
def construct_solution(pheromone, start_tau, iot_list, gb_list, uav_list):
    n = len(iot_list)
    #iot_map = {iot.id: iot for iot in iot_list}
    #iot_map = {iot.id: iot_list[i] for i in range(n)}  # id->IoT object, ids 1..n
    iot_map = {iot_list[i].id: iot_list[i] for i in range(n)}  # id -> IoT object

    all_unvisited = set(iot_map.keys())
    solution_uavs = {u.id: [('start', None, u.start)] for u in uav_list}

    for u in uav_list:
        cur_pos = u.start
        cur_time = 0
        prev_idx = None  # previous IoT index (0-based) for pheromone edge usage
        chosen_order = []  # keep IoT ids in the chosen order for this UAV (for 2-opt)
        # build a candidate list until no feasible IoT remains for this UAV
        while True:
            candidates = []
            for iid in list(all_unvisited):
                iobj = iot_map[iid]
                i_wp = point_on_line_at_distance_from_target(cur_pos, iobj.pos, R_IOT)
                # estimate times
                t_arrive = cur_time + travel_time(cur_pos, i_wp, u.V)
                t_capture = max(t_arrive, iobj.t_gen)
                nearest_g = min(gb_list, key=lambda g: dist(i_wp, g.pos))
                g_wp = point_on_line_at_distance_from_target(i_wp, nearest_g.pos, R_GBS)
                t_upload = t_capture + travel_time(i_wp, g_wp, u.V)
                # consider feasible if this upload can be done within Tmax
                if t_upload <= u.Tmax:
                    candidates.append((iid, i_wp, nearest_g.id, g_wp, t_upload))
            if not candidates:
                break

            # compute probs using edge pheromone or start_tau + AoI-aware heuristic
            probs = []
            denom = 0.0
            heur_vals = []
            tau_vals = []
            for (iid, i_wp, gid, g_wp, t_upload) in candidates:
                cur_idx = iid - 1
                # tau on edge prev->cur (or start->cur if prev_idx is None)
                if prev_idx is None:
                    tau = start_tau[cur_idx]
                else:
                    tau = pheromone[prev_idx, cur_idx]
                heur = heuristic_value(cur_pos, i_wp, cur_time, iot_map[iid], gb_list, u.V)
                val = (tau ** ALPHA) * (heur ** BETA)
                probs.append(val)
                denom += val
                heur_vals.append(heur)
                tau_vals.append(tau)
            if denom <= 0:
                break
            r = random.random() * denom
            acc = 0.0
            sel_idx = 0
            for k, val in enumerate(probs):
                acc += val
                if acc >= r:
                    sel_idx = k
                    break
            iid, i_wp, gid, g_wp, t_upload = candidates[sel_idx]

            # append IoT waypoint
            solution_uavs[u.id].append(('iot', iid, i_wp))
            chosen_order.append(iid)

            # decide whether to append GB immediately based on distance threshold
            if dist(i_wp, g_wp) < GB_IMMEDIATE_INSERT_THRESHOLD:
                solution_uavs[u.id].append(('gb', gid, g_wp))
                # update cur_pos to GB waypoint
                cur_pos = g_wp
            else:
                # do NOT append GB now; UAV stays at i_wp (pending capture)
                cur_pos = i_wp

            # cur_time must be set to the estimated upload time because subsequent feasibility checks used t_upload
            # NOTE: this is an estimate: if we didn't append GB we still use t_upload as conservative estimate
            cur_time = int(t_upload)
            # update prev_idx for pheromone edge usage: set prev to selected IoT index
            prev_idx = iid - 1
            # remove from unvisited
            all_unvisited.discard(iid)

        # after finishing selection for this UAV, apply a quick local 2-opt on the chosen IoT order
        if len(chosen_order) > 2:
            pos_map = {iid: iot_map[iid].pos for iid in chosen_order}
            improved_order = two_opt_order(chosen_order, pos_map)
            # rebuild path for this UAV from improved order
            rebuilt = [('start', None, u.start)]
            prev_pos = u.start
            for iid in improved_order:
                iobj = iot_map[iid]
                i_wp = point_on_line_at_distance_from_target(prev_pos, iobj.pos, R_IOT)
                nearest_g = min(gb_list, key=lambda g: dist(i_wp, g.pos))
                g_wp = point_on_line_at_distance_from_target(i_wp, nearest_g.pos, R_GBS)
                rebuilt.append(('iot', iid, i_wp))
                if dist(i_wp, g_wp) < GB_IMMEDIATE_INSERT_THRESHOLD:
                    rebuilt.append(('gb', nearest_g.id, g_wp))
                    prev_pos = g_wp
                else:
                    prev_pos = i_wp
            rebuilt.append(('end', None, u.end))
            solution_uavs[u.id] = clean_path_remove_consecutive_gbs(rebuilt, u)
        else:
            solution_uavs[u.id].append(('end', None, u.end))
            solution_uavs[u.id] = clean_path_remove_consecutive_gbs(solution_uavs[u.id], u)

    # fallback: assign unvisited to nearest UAVs (should be rare)
    if all_unvisited:
        for iid in list(all_unvisited):
            iobj = iot_map[iid]
            best_u = min(uav_list, key=lambda uu: dist(solution_uavs[uu.id][-2][2], iobj.pos))
            prev_pos = solution_uavs[best_u.id][-2][2]
            i_wp = point_on_line_at_distance_from_target(prev_pos, iobj.pos, R_IOT)
            nearest_g = min(gb_list, key=lambda g: dist(i_wp, g.pos))
            g_wp = point_on_line_at_distance_from_target(i_wp, nearest_g.pos, R_GBS)
            solution_uavs[best_u.id] = solution_uavs[best_u.id][:-1] + [('iot', iid, i_wp), ('gb', nearest_g.id, g_wp), ('end', None, best_u.end)]
            all_unvisited.discard(iid)

    return solution_uavs

# ----------------------------
# Compute fitness & AoI reward for pheromone deposit
# ----------------------------
def compute_fitness_for_solution(solution_uavs, iot_objs, gb_objs, base_uavs):
    uav_copies = [u.copy() for u in base_uavs]
    for u in uav_copies:
        u.path = solution_uavs[u.id]
    # finalize pending uploads (append nearest GBS before end if needed)
    for u in uav_copies:
        V_D, V_U, plen, tfin, feas = evaluate_uav_path(u, iot_objs, gb_objs)
        pending = [iid for iid in V_D if V_D[iid] is not None and V_U[iid] is None]
        if pending:
            nearest_g = min(gb_objs, key=lambda g: dist(u.path[-2][2], g.pos))
            g_wp = point_on_line_at_distance_from_target(u.path[-2][2], nearest_g.pos, R_GBS)
            u.path = u.path[:-1] + [('gb', nearest_g.id, g_wp)] + [u.path[-1]]
    V_D_glob, V_U_glob, per_uav_lengths, per_uav_finish_times, feas = evaluate_all_uavs(uav_copies, iot_objs, gb_objs)
    aoi_dict, peak, avg, coverage = compute_aoi_from_VU(V_U_glob, iot_objs)
    # reward: we want higher reward for *lower* avg AoI and higher coverage.
    # We normalize average AoI by T_MAX and form reward = (coverage) * (1 / (1 + avg_norm))
    avg_val = float('inf') if avg is None else float(avg)
    avg_norm = (avg_val / float(T_MAX)) if avg_val != float('inf') else 1.0
    coverage = coverage if coverage is not None else 0.0
    reward = coverage * (1.0 / (1.0 + avg_norm))  # between 0 and 1 roughly; higher is better



    ## Prepare objective values
    #peak_val = float('inf') if peak is None else float(peak)
    #avg_val = float('inf') if avg is None else float(avg)
    #total_dist = sum(per_uav_lengths.values()) if per_uav_lengths else 0.0

    ## Normalization
    #max_aoi_bound = float(T_MAX) if T_MAX > 0 else 1.0
    #peak_norm = (peak_val / max_aoi_bound) if peak_val != float('inf') else 1.0
    #avg_norm = (avg_val / max_aoi_bound) if avg_val != float('inf') else 1.0
    #max_dist_bound = len(uav_copies) * math.hypot(AREA_SIZE, AREA_SIZE)
    #dist_norm = total_dist / max_dist_bound if max_dist_bound > 0 else 0.0
    #coverage_norm = coverage if coverage is not None else 0.0

    ## infeasible penalty
    #infeasible_penalty = 0.0
    #for u in uav_copies:
    #    if not feas[u.id]:
    #        infeasible_penalty += 10000.0
    #w_peak = 0.45
    #w_avg = 0.15
    #w_dist = 0.20
    #w_cov = 0.40
    ## scalar score (smaller is better); coverage reduces score
    #score = (w_peak * peak_norm) + (w_avg * avg_norm) + (w_dist * dist_norm) - (w_cov * coverage_norm)
    #reward = score


    return reward, peak, avg, coverage, sum(per_uav_lengths.values()), per_uav_finish_times, feas, uav_copies

# ----------------------------
# ACO main loop with proper edge pheromone usage and elitist deposit
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
        for u in uav_list:
            prev_idx = None
            for typ, idx, pos in greedy_sol[u.id]:
                if typ == 'iot':
                    cur_idx = int(idx) - 1
                    if prev_idx is None:
                        start_tau[cur_idx] += 5.0  # boost start->node
                    else:
                        pheromone[prev_idx, cur_idx] += 5.0
                    prev_idx = cur_idx
                elif typ == 'start':
                    prev_idx = None

    best_solution = None
    best_reward = -1.0
    best_info = None
    history = []
    no_improve = 0

    for it in range(ACO_ITERS):
        ants_solutions = []
        ants_rewards = []

        for a in range(NUM_ANTS):
            sol = construct_solution(pheromone, start_tau, iot_objs, gb_objs, uav_list)
            reward, peak, avg, coverage, y_total, per_uav_finish_times, feas, uav_copies = compute_fitness_for_solution(sol, iot_objs, gb_objs, uav_list)
            ants_solutions.append((sol, reward, peak, avg, coverage, y_total, per_uav_finish_times, feas, uav_copies))
            ants_rewards.append(reward)
            if reward > best_reward:
                best_reward = reward
                best_solution = sol
                best_info = (reward, peak, avg, coverage, y_total, per_uav_finish_times, feas, uav_copies)
                no_improve = 0

        # if no ant improved best_reward, increment counter
        if all(r <= best_reward for r in ants_rewards):
            no_improve += 1
        else:
            no_improve = 0

        # evaporation
        pheromone *= (1.0 - RHO)
        start_tau *= (1.0 - RHO)

        # deposit from each ant (proportional to reward)
        delta = np.zeros_like(pheromone)
        delta_start = np.zeros_like(start_tau)
        for sol, reward, *_ in ants_solutions:
            deposit = LAMBDA * reward
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

        # elitist deposit: reinforce best-so-far path strongly
        if best_solution is not None:
            best_deposit = ELITIST_BOOST * (best_reward if best_reward > 0 else 1e-6)
            for u in uav_list:
                prev_idx = None
                for typ, idx, pos in best_solution[u.id]:
                    if typ == 'iot':
                        cur_idx = int(idx) - 1
                        if prev_idx is None:
                            delta_start[cur_idx] += best_deposit
                        else:
                            delta[prev_idx, cur_idx] += best_deposit
                        prev_idx = cur_idx
                    elif typ == 'start':
                        prev_idx = None

        # apply deposits
        pheromone += delta
        start_tau += delta_start
        # bounds
        np.clip(pheromone, 1e-9, 1e9, out=pheromone)
        np.clip(start_tau, 1e-9, 1e9, out=start_tau)

        if it % 10 == 0 or it == ACO_ITERS - 1:
            if best_info is None:
                print(f"Iter {it}: no solution yet")
            else:
                print(f"Iter {it}: best_reward={best_reward:.6f}, best_peak={best_info[1]}, best_avg={best_info[2]}, cov={best_info[3]:.3f}")

        history.append((it, best_reward, best_info))

        if no_improve >= NO_IMPROVE_LIMIT:
            print(f"No improvement for {NO_IMPROVE_LIMIT} iters -> early stop at iter {it}")
            break

    return best_solution, best_reward, best_info, history

# ----------------------------
# Build random scenario (same as before)
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
# ----------------------------
if __name__ == '__main__':
    iot_objs, gb_objs, uav_list = build_random_scenario(num_iot=NUM_IOT, num_gbs=NUM_GBS, num_uavs=NUM_UAV, seed=SEED)
    print(f"Scenario: IoTs={len(iot_objs)}, GBSs={len(gb_objs)}, UAVs={len(uav_list)}")
    t0 = time.time()
    best_sol, best_reward, best_info, history = run_aco(iot_objs, gb_objs, uav_list)
    elapsed = time.time() - t0
    print(f"ACO done in {elapsed:.2f}s: best_reward={best_reward:.6f}")
    if best_info is not None:
        print(f"Best info (reward, peak, avg, coverage): {best_info[0]:.6f}, {best_info[1]}, {best_info[2]}, {best_info[3]:.3f}")

    if best_sol is not None:
        _, peak, avg, coverage, y_total, per_uav_finish_times, feas_flags, uav_copies = compute_fitness_for_solution(best_sol, iot_objs, gb_objs, uav_list)
        V_D_glob, V_U_glob, per_uav_lengths, per_uav_finish_times, feas_flags = evaluate_all_uavs(uav_copies, iot_objs, gb_objs)
        aoi_dict, peak, avg, coverage = compute_aoi_from_VU(V_U_glob, iot_objs)
        print("Per-UAV finish times:", per_uav_finish_times)
        print("Per-UAV feasible:", feas_flags)
        print(f"Peak AoI: {peak}, Avg AoI: {avg}, Coverage: {coverage:.3f}")

        # print paths
        for u in uav_copies:
            print(f"UAV{u.id} path:")
            for typ, idx, pos in u.path:
                print(f"  {typ:5s} {str(idx):>4s} @ ({pos[0]:.2f},{pos[1]:.2f})")
            print("")
        # plot
        plot_solution(iot_objs, gb_objs, uav_copies, title="ACO improved (range-aware)")
    else:
        print("No feasible solution produced by ACO.")
