# ga_multi_uav_im_with_gbs.py
"""
GA-based solver for the multi-UAV IoT collection problem (range-aware).
This version inserts GBS waypoints after IoT waypoints (range-aware), similar to the greedy heuristic.
Objectives combined into scalar fitness:
 - minimize peak AoI
 - minimize average AoI
 - minimize total UAV travel distance
 - maximize coverage

Run as a script. Requires: numpy, matplotlib
"""
import random
import copy
import math
import time
from typing import List
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------
AREA_SIZE = 500  # map is AREA_SIZE x AREA_SIZE units
UAV_SPEED = 10.0  # units per time slot
T_MAX = 250  # max allowed time per UAV (time slots)
R_IOT = 10.0  # IoT communication range (units)
R_GBS = 30.0  # GBS communication range (units)
NUM_UAV = 3
NUM_GBS = 4
NUM_IOT = 50  # change this variable to run experiments for 10,20,...
SEED = 42

# GA hyperparameters
POP_SIZE = 60
GENERATIONS = 120
TOURNAMENT_K = 3
CX_RATE = 0.8
MUT_RATE = 0.15
ELITE_K = 2

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
# Evaluation functions
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
# Route builder from assignment (now inserts GBS after each IoT waypoint)
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
# Fitness function (multi-objective scalarization)
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
    score = (w_peak * peak_norm) + (w_avg * avg_norm) + (w_dist * dist_norm) - (w_cov * coverage_norm) + infeasible_penalty

    return score, peak_val, avg_val, coverage_norm, total_dist, per_uav_finish_times, feas, uav_copies

# ----------------------------
# GA operators
# ----------------------------
def rand_assignment(num_iot, num_uav):
    return [random.randrange(num_uav) for _ in range(num_iot)]

def tournament_selection(pop, fits, k=TOURNAMENT_K):
    best_idx = None
    for _ in range(k):
        i = random.randrange(len(pop))
        if best_idx is None or fits[i] < fits[best_idx]:
            best_idx = i
    return copy.deepcopy(pop[best_idx])

def uniform_crossover(a, b):
    n = len(a)
    child1 = a[:]
    child2 = b[:]
    for i in range(n):
        if random.random() < 0.5:
            child1[i], child2[i] = child2[i], child1[i]
    return child1, child2

def mutate_assignment(chrom, num_uav, mut_rate=MUT_RATE):
    n = len(chrom)
    for i in range(n):
        if random.random() < mut_rate:
            chrom[i] = random.randrange(num_uav)
    return chrom

# ----------------------------
# GA main loop
# ----------------------------
def run_ga(iot_objs, gb_objs, base_uavs, pop_size=POP_SIZE, generations=GENERATIONS):
    num_iot = len(iot_objs)
    num_uav = len(base_uavs)
    random.seed(SEED)
    np.random.seed(SEED)

    # initial population (random)
    pop = [rand_assignment(num_iot, num_uav) for _ in range(pop_size)]
    fits = []
    info = []
    for ind in pop:
        score, peak, avg_aoi, coverage, total_dist, per_uav_finish_times, feas, _ = evaluate_assignment_fitness(ind, iot_objs, gb_objs, base_uavs)
        fits.append(score)
        info.append((peak, avg_aoi, coverage, total_dist, per_uav_finish_times, feas))

    best_history = []
    for gen in range(generations):
        new_pop = []
        # elitism: carry best K
        sorted_idx = sorted(range(len(pop)), key=lambda i: fits[i])
        for k in range(ELITE_K):
            new_pop.append(copy.deepcopy(pop[sorted_idx[k]]))
        # generate rest
        while len(new_pop) < pop_size:
            # selection
            p1 = tournament_selection(pop, fits)
            p2 = tournament_selection(pop, fits)
            # crossover
            if random.random() < CX_RATE:
                c1, c2 = uniform_crossover(p1, p2)
            else:
                c1, c2 = p1, p2
            # mutate
            c1 = mutate_assignment(c1, num_uav)
            c2 = mutate_assignment(c2, num_uav)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        pop = new_pop
        # re-evaluate
        fits = []
        info = []
        best_idx = None
        for i, ind in enumerate(pop):
            score, peak, avg_aoi, coverage, total_dist, per_uav_finish_times, feas, _ = evaluate_assignment_fitness(ind, iot_objs, gb_objs, base_uavs)
            fits.append(score)
            info.append((peak, avg_aoi, coverage, total_dist, per_uav_finish_times, feas))
            if best_idx is None or score < fits[best_idx]:
                best_idx = i
        best_history.append((gen, fits[best_idx], info[best_idx]))
        if gen % 10 == 0 or gen == generations - 1:
            bp = info[best_idx]
            print(f"Gen {gen}: best_score={fits[best_idx]:.4f}, peak={bp[0]}, avg={bp[1]}, coverage={bp[2]:.3f}, total_dist={bp[3]:.1f}")
    best_idx = int(np.argmin(fits))
    return pop[best_idx], fits[best_idx], info[best_idx], best_history

# ----------------------------
# Scenario generation and plotting
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
# Main driver
# ----------------------------
if __name__ == '__main__':
    iot_objs, gb_list, uav_list = build_random_scenario(num_iot=NUM_IOT, num_gbs=NUM_GBS, num_uavs=NUM_UAV, seed=SEED)
    print(f"Scenario: IoTs={len(iot_objs)}, GBSs={len(gb_list)}, UAVs={len(uav_list)}")
    t0 = time.time()
    best_chrom, best_score, best_info, history = run_ga(iot_objs, gb_list, uav_list)
    elapsed = time.time() - t0
    print(f"GA done in {elapsed:.2f}s: best_score={best_score:.4f}")
    bp = best_info
    print(f"Best info (peak, avg, coverage, total_dist): {bp[0]}, {bp[1]}, {bp[2]}, {bp[3]:.1f}")

    # reconstruct best solution paths
    score, peak, avg_aoi, coverage, total_dist, per_uav_finish_times, feas_flags, uav_copies = evaluate_assignment_fitness(best_chrom, iot_objs, gb_list, uav_list)
    V_D_glob, V_U_glob, per_uav_lengths, per_uav_finish_times, feas_flags = evaluate_all_uavs(uav_copies, iot_objs, gb_list)
    aoi_dict, peak, avg, coverage = compute_aoi_from_VU(V_U_glob, iot_objs)
    print("Per-UAV finish times:", per_uav_finish_times)
    print("Per-UAV feasible:", feas_flags)
    print(f"Peak AoI: {peak}, Avg AoI: {avg}, Coverage: {coverage:.3f}")

    for iid in sorted(aoi_dict.keys()):
        print(f" IoT {iid}: AoI = {aoi_dict[iid]}  (t_gen={next(i.t_gen for i in iot_objs if i.id == iid)})")

    # print paths
    for u in uav_copies:
        print(f"UAV{u.id} path:")
        for typ, idx, pos in u.path:
            print(f"  {typ:5s} {str(idx):>4s} @ ({pos[0]:.2f},{pos[1]:.2f})")
        print("")

    served_iots = {iot_id for iot_id, uav_id in V_U_glob.items() if uav_id is not None}
    #print("Served IoTs:", served_iots)
    all_ids = set(u.id for u in iot_objs)
    served_ids = set(served_iots)
    not_served_ids = all_ids - served_ids
    print(f"   Not served iots:{not_served_ids}, total: {len(not_served_ids)}")
    # plot
    plot_solution(iot_objs, gb_list, uav_copies, title="GA multi-UAV (range-aware) result")
