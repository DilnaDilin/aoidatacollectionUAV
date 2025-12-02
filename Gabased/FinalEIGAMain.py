#!/usr/bin/env python3
"""
ga_multi_uav_eipga_full.py

Complete script integrating:
 - Original range-aware GA evaluator for multi-UAV IoT data collection (AoI + GBS uploads)
 - New EIPGA variant using Path-Breakpoint encoding (perm + breakpoints)
   with Flip, Swap, LSlide, RSlide mutation ops, breakpoint updates,
   mutation-before-selection, and hybrid selection (0.7 elite, 0.2 roulette, 0.1 random).

Save and run: python ga_multi_uav_eipga_full.py
Requires: numpy, matplotlib
"""
import random
import copy
import math
import time
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameters (tune as needed)
# ----------------------------
AREA_SIZE = 500  # map is AREA_SIZE x AREA_SIZE units
UAV_SPEED = 10.0  # units per time slot
T_MAX = 250  # max allowed time per UAV (time slots)
R_IOT = 10.0  # IoT communication range (units)
R_GBS = 30.0  # GBS communication range (units)
NUM_UAV = 3
NUM_GBS = 4
NUM_IOT = 60  # number of IoT nodes (DCPs)
SEED = 42

# EIPGA hyperparameters
EIPGA_POP = 30      # Np
EIPGA_GEN = 50     # Ng
EIPGA_NNEW = 120    # Nnew (offspring per generation)
EIPGA_VERBOSE = True

# ----------------------------
# Utilities (original)
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
    """Remove consecutive GB waypoints; ensure start and end are present"""
    cleaned = []
    for entry in path:
        if cleaned and cleaned[-1][0] == 'gb' and entry[0] == 'gb':
            # If same type 'gb' consecutively — skip the redundant one
            continue
        cleaned.append(entry)
    # ensure start and end are correct
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
# Evaluation functions (original)
# ----------------------------
def evaluate_uav_path(uav, iot_objs, gb_objs, R_I=R_IOT, R_G=R_GBS):
    """
    Simulate UAV path (waypoints already contain positions where UAV moves).
    Returns:
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
            # wait until generation if arrived earlier
            if t < iot_tgen[idx]:
                wait = int(iot_tgen[idx] - t)
                t += wait
            V_D[idx] = int(t)
            captured_pending.append(idx)
        elif typ == 'gb':
            # upload all pending
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
# Route builder from assignment (original)
# ----------------------------
def build_routes_from_assignment(assign: List[int], iot_objs: List[IoT], gb_objs: List[GBS], uav_list: List[UAV], R_I=R_IOT, R_G=R_GBS):
    """
    Given an assignment vector assign[i]=u (u in 0..num_uav-1), build UAV paths.
    For each assigned IoT we place:
      - an IoT waypoint (range-aware) computed from the previous UAV waypoint
      - immediately after, a GBS waypoint (nearest GBS range boundary) so the UAV uploads
    We then clean consecutive GBS waypoints.
    """
    # collect assigned IoTs per UAV (UAV ids are 1-based)
    assigned = {u.id: [] for u in uav_list}
    for idx, gene in enumerate(assign):
        uidx = int(gene)
        u_id = uidx + 1
        if u_id in assigned:
            assigned[u_id].append(iot_objs[idx].id)

    routes = {u.id: [('start', None, u.start)] for u in uav_list}

    # Precompute IoT map for quick lookup
    iot_map = {iot.id: iot for iot in iot_objs}
    # For GBS nearest, we will use gb_objs list directly

    for u in uav_list:
        cur_pos = u.start
        # simple nearest-neighbor ordering for assigned IoTs
        remaining = assigned[u.id][:]
        order = []
        while remaining:
            best = min(remaining, key=lambda iid: dist(cur_pos, iot_map[iid].pos))
            order.append(best)
            cur_pos = iot_map[best].pos
            remaining.remove(best)
        # build path: for each iot in order, compute i_wp (from previous waypoint) and then g_wp (nearest GBS)
        prev_pos = u.start
        for iid in order:
            iobj = iot_map[iid]
            i_wp = point_on_line_at_distance_from_target(prev_pos, iobj.pos, R_I)
            # choose nearest GBS (in terms of travel from i_wp)
            nearest_g = min(gb_objs, key=lambda g: dist(i_wp, g.pos))
            g_wp = point_on_line_at_distance_from_target(i_wp, nearest_g.pos, R_G)
            routes[u.id].append(('iot', iid, i_wp))
            if dist(i_wp, g_wp) < 2*R_GBS:
                routes[u.id].append(('gb', nearest_g.id, g_wp))
                prev_pos = g_wp  # UAV will continue from after upload
        routes[u.id].append(('end', None, u.end))
        # clean consecutive gb waypoints (if any) and ensure start/end correctness
        routes[u.id] = clean_path_remove_consecutive_gbs(routes[u.id], u)

    # convert to UAV.path list objects
    uav_paths = {}
    for u in uav_list:
        uav_paths[u.id] = routes[u.id]
    return uav_paths

# ----------------------------
# Fitness function (original)
# ----------------------------
def evaluate_assignment_fitness(assign: List[int],
                                iot_objs,
                                gb_objs,
                                base_uavs,
                                Tmax_norm=T_MAX,
                                area_size=AREA_SIZE,
                                w_peak=0.45,
                                w_avg=0.15,
                                w_dist=0.20,
                                w_cov=0.40):
    """
    Build routes from assignment, finalize uploads (fallback), evaluate globally and
    compute scalar fitness combining peak AoI, avg AoI, total distance, coverage.
    Returns:
      score, peak_val, avg_val, coverage_norm, total_dist, per_uav_finish_times, feasibility_flags, uav_copies
    """
    # Build uav copies and assign paths
    uav_copies = [u.copy() for u in base_uavs]
    uav_paths = build_routes_from_assignment(assign, iot_objs, gb_objs, uav_copies)
    for u in uav_copies:
        u.path = uav_paths[u.id]

    # finalize: ensure uploads by adding GBS if needed (fallback)
    for u in uav_copies:
        V_D, V_U, plen, tfin, feas = evaluate_uav_path(u, iot_objs, gb_objs)
        pending = [iid for iid in V_D if V_D[iid] is not None and V_U[iid] is None]
        if pending:
            nearest_g = min(gb_objs, key=lambda g: dist(u.path[-2][2], g.pos))
            g_wp = point_on_line_at_distance_from_target(u.path[-2][2], nearest_g.pos, R_G)
            u.path = u.path[:-1] + [('gb', nearest_g.id, g_wp)] + [u.path[-1]]

    # evaluate globally
    V_D_glob, V_U_glob, per_uav_lengths, per_uav_finish_times, feas = evaluate_all_uavs(uav_copies, iot_objs, gb_objs)
    aoi_dict, peak, avg, coverage = compute_aoi_from_VU(V_U_glob, iot_objs)

    # Prepare objective values
    peak_val = float('inf') if peak is None else float(peak)
    avg_val = float('inf') if avg is None else float(avg)
    total_dist = sum(per_uav_lengths.values()) if per_uav_lengths else 0.0

    # Normalization
    max_aoi_bound = float(Tmax_norm) if Tmax_norm > 0 else 1.0
    peak_norm = (peak_val / max_aoi_bound) if peak_val != float('inf') else 1.0
    avg_norm = (avg_val / max_aoi_bound) if avg_val != float('inf') else 1.0
    max_dist_bound = len(uav_copies) * math.hypot(area_size, area_size)
    dist_norm = total_dist / max_dist_bound if max_dist_bound > 0 else 0.0
    coverage_norm = coverage if coverage is not None else 0.0

    # infeasible penalty
    infeasible_penalty = 0.0
    for u in uav_copies:
        if not feas[u.id]:
            infeasible_penalty += 10000.0

    # scalar score (smaller is better); coverage reduces score
    #score = (w_peak * peak_norm) + (w_avg * avg_norm) + (w_dist * dist_norm) - (w_cov * coverage_norm) + infeasible_penalty
    score = (w_avg * avg_norm) + infeasible_penalty

    return score, peak_val, avg_val, coverage_norm, total_dist, per_uav_finish_times, feas, uav_copies

# ----------------------------
# Scenario generation and plotting (original)
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

def plot_solution(iot_list, gb_list, uav_list, title="GA multi-UAV solution"):
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
# EIPGA implementation (IoT == DCP)
# ----------------------------
def init_chromosome_path_breakpoint(num_iot: int, num_uav: int):
    """Return (perm, breakpoints)
       perm: list of IoT indices [0..num_iot-1]
       breakpoints: sorted list length num_uav-1 with values in [1..num_iot-1]
    """
    perm = list(range(num_iot))
    random.shuffle(perm)
    if num_uav <= 1:
        bps = []
    else:
        candidates = list(range(1, num_iot))
        # if num_iot < num_uav: allow duplicates? better to create increasing positions evenly
        if len(candidates) >= (num_uav - 1):
            bps = sorted(random.sample(candidates, num_uav - 1))
        else:
            # too few positions: create near-even breakpoints
            bps = []
            for k in range(1, num_uav):
                pos = int(round(k * num_iot / num_uav))
                pos = max(1, min(num_iot - 1, pos))
                bps.append(pos)
            bps = sorted(list(set(bps)))[:max(0, num_uav - 1)]
            # if too short, pad
            while len(bps) < (num_uav - 1):
                bps.append(min(num_iot - 1, (bps[-1] + 1) if bps else 1))
            bps = sorted(bps)
    return perm, bps

def decode_chromosome_to_assignment(chrom: Tuple[List[int], List[int]], num_uav: int, num_iot: int):
    """Convert (perm, bps) to assignment vector of length num_iot with values 0..num_uav-1"""
    perm, bps = chrom
    segments = []
    start = 0
    for bp in bps:
        segments.append(perm[start:bp])
        start = bp
    segments.append(perm[start:])
    # make sure number of segments equals num_uav
    if len(segments) < num_uav:
        # append empty segments
        for _ in range(num_uav - len(segments)):
            segments.append([])
    elif len(segments) > num_uav:
        # merge extra suffix into last segment
        extra = segments[num_uav:]
        segments = segments[:num_uav]
        for seg in extra:
            segments[-1].extend(seg)
    assign = [None] * num_iot
    for uidx, seg in enumerate(segments):
        for iot_idx in seg:
            assign[iot_idx] = uidx
    # If some IoT not assigned (shouldn't happen), assign randomly
    for i in range(num_iot):
        if assign[i] is None:
            assign[i] = random.randrange(num_uav)
    return assign

# Mutation operators on permutation
def op_swap(perm):
    if len(perm) < 2:
        return
    a, b = random.sample(range(len(perm)), 2)
    perm[a], perm[b] = perm[b], perm[a]

def op_flip(perm):
    if len(perm) < 2:
        return
    a, b = sorted(random.sample(range(len(perm)), 2))
    perm[a:b+1] = list(reversed(perm[a:b+1]))

def op_lslide(perm):
    n = len(perm)
    if n < 2:
        return
    i = random.randrange(n)
    if i == 0:
        return
    val = perm.pop(i)
    perm.insert(max(0, i-1), val)

def op_rslide(perm):
    n = len(perm)
    if n < 2:
        return
    i = random.randrange(n)
    if i == n-1:
        return
    val = perm.pop(i)
    perm.insert(min(len(perm), i+1), val)

def breakpoint_update(bps, num_iot, num_uav):
    """Adjust breakpoints by -1,0,+1 randomly while maintaining order and bounds"""
    if num_uav <= 1:
        return []
    bps = sorted(bps[:])  # copy
    for i in range(len(bps)):
        delta = random.choice([-1, 0, 1])
        newv = bps[i] + delta
        low = 1 if i == 0 else bps[i-1] + 1
        high = num_iot - (len(bps) - i)
        newv = max(low, min(high, newv))
        bps[i] = newv
    # ensure strictly increasing
    bps = sorted(bps)
    for i in range(1, len(bps)):
        if bps[i] <= bps[i-1]:
            bps[i] = bps[i-1] + 1
    # final clipping
    for i in reversed(range(len(bps))):
        if bps[i] >= num_iot:
            bps[i] = num_iot - (len(bps) - i)
    return bps

def mutate_chromosome(chrom, num_iot, num_uav, pmut_perm=0.8, p_bp_mut=0.6):
    """Apply multiple mutation ops to the chromosome (partheno-genetic style)."""
    perm, bps = chrom
    perm = perm[:]  # copy
    bps = bps[:]    # copy
    nops = random.choice([1,1,2,2,3])  # bias toward 1-2 ops
    ops = [op_swap, op_flip, op_lslide, op_rslide]
    for _ in range(nops):
        if random.random() < pmut_perm:
            op = random.choice(ops)
            op(perm)
    if random.random() < p_bp_mut:
        bps = breakpoint_update(bps, num_iot, num_uav)
    return perm, bps

# hybrid selection operator
def hybrid_selection_from_candidates(candidates, cand_fits, Np):
    """candidates: list of chromosomes (perm, bps)
       cand_fits: list of fitness scores (smaller is better)
    """
    Np = int(Np)
    n_elite = int(round(0.7 * Np))
    n_roulette = int(round(0.2 * Np))
    n_random = Np - (n_elite + n_roulette)
    # sort by fitness (ascending)
    idx_sorted = sorted(range(len(candidates)), key=lambda i: cand_fits[i])
    new_pop = []
    # elites
    for i in idx_sorted[:n_elite]:
        new_pop.append(copy.deepcopy(candidates[i]))
    # roulette: weight = maxf - f (+eps)
    maxf = max(cand_fits)
    eps = 1e-8
    weights = [(maxf - f) + eps for f in cand_fits]
    if sum(weights) <= eps * len(weights) * 1.1:
        weights = None
    else:
        s = sum(weights)
        weights = [w / s for w in weights]
    for _ in range(n_roulette):
        if weights is None:
            i = random.randrange(len(candidates))
        else:
            i = int(np.random.choice(range(len(candidates)), p=weights))
        new_pop.append(copy.deepcopy(candidates[i]))
    # random/new: generate fresh chromosomes
    num_iot = len(candidates[0][0])
    inferred_num_uav = len(candidates[0][1]) + 1
    for _ in range(n_random):
        new_pop.append(init_chromosome_path_breakpoint(num_iot, inferred_num_uav))
    # adjust length
    while len(new_pop) < Np:
        new_pop.append(copy.deepcopy(candidates[idx_sorted[0]]))
    if len(new_pop) > Np:
        new_pop = new_pop[:Np]
    return new_pop

def evaluate_chromosome_fitness(chrom, iot_objs, gb_objs, base_uavs):
    """ Decode chrom to assignment and call existing evaluate_assignment_fitness. """
    num_iot = len(iot_objs)
    num_uav = len(base_uavs)
    assign = decode_chromosome_to_assignment(chrom, num_uav, num_iot)
    score, peak_val, avg_val, coverage_norm, total_dist, per_uav_finish_times, feas, uav_copies = evaluate_assignment_fitness(assign, iot_objs, gb_objs, base_uavs)
    return score, peak_val, avg_val, coverage_norm, total_dist, per_uav_finish_times, feas, uav_copies

def run_eipga(iot_objs, gb_objs, base_uavs, Np=EIPGA_POP, Ng=EIPGA_GEN, Nnew=EIPGA_NNEW, seed=SEED, verbose=EIPGA_VERBOSE):
    """
    EIPGA main loop.
    """
    random.seed(seed)
    np.random.seed(seed)
    num_iot = len(iot_objs)
    num_uav = len(base_uavs)
    if Nnew is None:
        Nnew = 2 * Np

    # init population
    pop = [init_chromosome_path_breakpoint(num_iot, num_uav) for _ in range(Np)]

    # initial eval (for tracking)
    fits = []
    info = []
    for chrom in pop:
        score, peak, avg_aoi, coverage, total_dist, per_uav_finish_times, feas, _ = evaluate_chromosome_fitness(chrom, iot_objs, gb_objs, base_uavs)
        fits.append(score)
        info.append((peak, avg_aoi, coverage, total_dist, per_uav_finish_times, feas))

    best_overall_idx = int(np.argmin(fits)) if fits else 0
    best_overall = (pop[best_overall_idx], fits[best_overall_idx], info[best_overall_idx]) if fits else (pop[0], float('inf'), None)
    best_history = []

    for gen in range(Ng):
        # generate Nnew offspring by mutation-before-selection
        candidates = []
        for _ in range(Nnew):
            parent = random.choice(pop)
            child = mutate_chromosome(parent, num_iot, num_uav)
            candidates.append(child)
        # evaluate candidates
        cand_fits = []
        cand_info = []
        for chrom in candidates:
            score, peak, avg_aoi, coverage, total_dist, per_uav_finish_times, feas, _ = evaluate_chromosome_fitness(chrom, iot_objs, gb_objs, base_uavs)
            cand_fits.append(score)
            cand_info.append((peak, avg_aoi, coverage, total_dist, per_uav_finish_times, feas))
        # hybrid selection -> next population
        pop = hybrid_selection_from_candidates(candidates, cand_fits, Np)
        # evaluate new pop
        fits = []
        info = []
        best_idx = None
        for i, chrom in enumerate(pop):
            score, peak, avg_aoi, coverage, total_dist, per_uav_finish_times, feas, _ = evaluate_chromosome_fitness(chrom, iot_objs, gb_objs, base_uavs)
            fits.append(score)
            info.append((peak, avg_aoi, coverage, total_dist, per_uav_finish_times, feas))
            if best_idx is None or score < fits[best_idx]:
                best_idx = i
        # update best overall
        if fits and (best_overall[1] is None or fits[best_idx] < best_overall[1]):
            best_overall = (pop[best_idx], fits[best_idx], info[best_idx])
        best_history.append((gen, fits[best_idx], info[best_idx]))
        if verbose and (gen % 10 == 0 or gen == Ng - 1):
            bp = info[best_idx]
            print(f"EIPGA Gen {gen}: best_score={fits[best_idx]:.6f}, peak={bp[0]}, avg={bp[1]}, coverage={bp[2]:.3f}, total_dist={bp[3]:.1f}")

    best_chrom = best_overall[0]
    best_score = best_overall[1]
    best_info = best_overall[2]
    # decode best to assignment and evaluate final UAV copies
    assign = decode_chromosome_to_assignment(best_chrom, num_uav, num_iot)
    score, peak, avg, coverage, total_dist, per_uav_finish_times, feas, uav_copies = evaluate_assignment_fitness(assign, iot_objs, gb_objs, base_uavs)
    return best_chrom, best_score, best_info, best_history, assign, uav_copies

# ----------------------------
# Main driver



import datetime
import csv
import time

# ----------------------------
# Multi-seed experiment driver for EIPGA
# ----------------------------
def run_multi_seed_experiment_eipga(seeds_list, num_iot=NUM_IOT, num_gbs=NUM_GBS, num_uavs=NUM_UAV):
    results = []

    for s in seeds_list:
        print("\n==============================")
        print(f" Running EIPGA Experiment - Seed {s}")
        print("==============================")

        # Build scenario
        iot_objs, gb_objs, uav_list = build_random_scenario(num_iot=num_iot, num_gbs=num_gbs, num_uavs=num_uavs, seed=s)
        print(f"Scenario: IoTs={len(iot_objs)}, GBSs={len(gb_objs)}, UAVs={len(uav_list)}")

        # Run EIPGA
        t0 = time.time()
        best_chrom, best_score, best_info, history, best_assign, uav_copies = run_eipga(
            iot_objs, gb_objs, uav_list, Np=EIPGA_POP, Ng=EIPGA_GEN, Nnew=EIPGA_NNEW, seed=s, verbose=True
        )
        elapsed = time.time() - t0
        print(f"EIPGA done in {elapsed:.2f}s: best_score={best_score:.6f}")

        # Evaluate final solution
        score, peak, avg, coverage, total_dist, per_uav_finish_times, feas_flags, uav_copies = evaluate_assignment_fitness(
            best_assign, iot_objs, gb_objs, uav_list
        )
        V_D_glob, V_U_glob, per_uav_lengths, per_uav_finish_times, feas_flags = evaluate_all_uavs(uav_copies, iot_objs, gb_objs)
        aoi_dict, peak, avg, coverage = compute_aoi_from_VU(V_U_glob, iot_objs)

        # Unserved IoTs
        served_iots = {iot_id for iot_id, v in V_U_glob.items() if v is not None}
        all_ids = set(iot.id for iot in iot_objs)
        not_served_ids = all_ids - served_iots

        print("Per-UAV finish times:", per_uav_finish_times)
        print("Per-UAV feasible:", feas_flags)
        print(f"Peak AoI: {peak}, Avg AoI: {avg}, Coverage: {coverage:.3f}")
        print(f"Not served IoTs: {not_served_ids}, total: {len(not_served_ids)}")

        results.append({
            "seed": s,
            "fitness": score,
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

    return results

# ----------------------------
# Compute aggregated stats
# ----------------------------
def compute_final_stats_eipga(results):
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
def save_results_csv_eipga(results, final_stats, num_iot=NUM_IOT, num_uavs=NUM_UAV, num_gbs=NUM_GBS):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eipga_multi_seed_results_{timestamp}.csv"

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

    results = run_multi_seed_experiment_eipga(seeds)
    final_stats = compute_final_stats_eipga(results)

    print("\n===== Aggregated Stats =====")
    for k, v in final_stats.items():
        print(f"{k}: {v}")

    save_results_csv_eipga(results, final_stats)
