# Greedy multi-UAV heuristic — paper-aligned version with IoT/GBS ranges
# Requirements: numpy, matplotlib
# Run as a script or in a notebook cell.

import math
import copy
import time
import random
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameters (tweak as needed)
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


# ----------------------------
# Utility functions
# ----------------------------
def dist(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return float(np.linalg.norm(a - b))


def travel_time(a, b, V):
    """Discrete-time travel: ceil(distance / V)"""
    d = dist(a, b)
    return math.ceil(d / V)


def point_on_line_at_distance_from_target(prev_pos, target_pos, radius):
    """
    Return a point on the line between prev_pos and target_pos that is at
    distance = radius from target_pos (i.e., the boundary point inside target's range
    along the approach from prev_pos). If prev_pos is already within radius,
    return prev_pos (no movement needed for capture/upload).
    """
    prev = np.array(prev_pos, dtype=float)
    tgt = np.array(target_pos, dtype=float)
    d = dist(prev, tgt)
    if d <= radius:
        return tuple(prev)  # already inside the circle
    # unit vector from target to prev: prev - tgt
    vec = prev - tgt
    if np.isclose(np.linalg.norm(vec), 0.0):
        # degenerate: prev == target; return target + radius along x
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
        # path: list of waypoints (type, id, (x,y))
        # types: 'start','iot','gb','end'
        self.path = [('start', None, self.start), ('end', None, self.end)]

    def copy(self):
        u = UAV(self.id, self.start, self.end, self.V, self.Tmax)
        u.path = copy.deepcopy(self.path)
        return u


# ----------------------------
# Simulation / evaluation
# ----------------------------
def evaluate_uav_path(uav, iot_objs, gb_objs, R_I=R_IOT, R_G=R_GBS):
    """
    Simulate UAV path (waypoints already contain positions where UAV moves).
    Returns:
      V_D: dict IoT id -> capture time (or None)
      V_U: dict IoT id -> upload time (or None)
      total_dist, finish_time, feasible (<=Tmax)
    Notes:
      - When visiting an 'iot' waypoint, we treat that waypoint as inside IoT range.
      - If arrived earlier than t_gen, UAV waits until t_gen (discrete time).
      - When visiting a 'gb' waypoint, all pending captured IoTs (by this UAV) are uploaded at that time.
      - If there are pending captures at the end, evaluator simulates flying to nearest GBS (GBS-range point) and uploads.
    """
    V_D = {iot.id: None for iot in iot_objs}
    V_U = {iot.id: None for iot in iot_objs}
    iot_tgen = {iot.id: iot.t_gen for iot in iot_objs}
    gb_map = {g.id: g.pos for g in gb_objs}

    cur_pos = np.array(uav.path[0][2], dtype=float)  # start
    t = 0
    total_dist = 0.0
    captured_pending = []  # iot ids captured but not uploaded

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

    # if pending, try to upload by flying to nearest GBS (at the GBS boundary point)
    if captured_pending:
        # nearest GBS index
        nearest_g = min(gb_objs, key=lambda g: dist(cur_pos, g.pos))
        # compute a point at distance R_G from GBS on line from current pos to GBS (so UAV reaches GBS range)
        g_w = point_on_line_at_distance_from_target(cur_pos, nearest_g.pos, R_G)
        d = dist(cur_pos, g_w)
        total_dist += d
        tt = travel_time(cur_pos, g_w, uav.V)
        t += tt
        # upload all pending now
        for iid in list(captured_pending):
            V_U[iid] = int(t)
            captured_pending.remove(iid)
    finish_time = int(t)
    feasible = finish_time <= uav.Tmax
    return V_D, V_U, total_dist, finish_time, feasible


def evaluate_all_uavs(uav_list, iot_objs, gb_objs):
    """
    Evaluate all UAVs and merge their V_D and V_U results.
    Returns:
      V_D_global, V_U_global, per_uav_lengths, per_uav_finish_times, feasibility_flags
    """
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
# Greedy heuristic helpers
# ----------------------------
def earliest_iot_delivery_for_uav(uav, Lu_cur_pos, Tcur, unvisited_iots, iot_objs, gb_objs, R_I=R_IOT, R_G=R_GBS):
    """
    Algorithm 1: For given UAV location Lu_cur_pos and time Tcur, find the IoT (from unvisited list)
    that can be delivered the earliest (capture + nearest-GBS upload), returning:
      (iot_id, gb_id, upload_time, capture_time, iot_waypoint, gb_waypoint)
    Waypoints are computed using the range-based intersection approach.
    """
    best = (None, None, float('inf'), None, None, None)
    # map id -> object
    iot_map = {iot.id: iot for iot in iot_objs}
    gb_map = {g.id: g for g in gb_objs}

    for i in unvisited_iots:
        iobj = iot_map[i]
        # compute IoT waypoint: move from Lu_cur_pos towards iot until within R_I of iot
        i_wp = point_on_line_at_distance_from_target(Lu_cur_pos, iobj.pos, R_I)
        # travel time to reach i_wp
        t_arrive = Tcur + travel_time(Lu_cur_pos, i_wp, uav.V)
        # capture time
        t_capture = max(t_arrive, iobj.t_gen)
        # find closest GBS for upload
        # compute GBS waypoint from i_wp towards each GBS
        best_g = None;
        best_upload_time = float('inf');
        best_g_wp = None
        for g in gb_objs:
            g_wp = point_on_line_at_distance_from_target(i_wp, g.pos, R_G)
            t_upload = t_capture + travel_time(i_wp, g_wp, uav.V)
            if t_upload < best_upload_time:
                best_upload_time = t_upload
                best_g = g.id
                best_g_wp = g_wp
        if best_upload_time < best[2]:
            best = (i, best_g, int(best_upload_time), int(t_capture), i_wp, best_g_wp)
    return best  # (iot_id, gb_id, upload_time, capture_time, i_wp, g_wp)


def try_insertion_global(uav_list, target_uav_id, iot_id, gb_id, iot_objs, gb_objs, R_I=R_IOT, R_G=R_GBS):
    """
    Try inserting IoT (with optional GBS) at all spots on target UAV's path.
    Evaluate each candidate by simulating all UAVs and choosing the candidate minimizing global peak AoI.
    Enforce feasibility (target UAV must remain within Tmax).
    Returns best_path (list of waypoints), best_peak (numeric), best_finish_time
    or (None, inf, None) if no feasible candidate.
    """
    target = next(u for u in uav_list if u.id == target_uav_id)
    orig = target.path
    P_len = len(orig)
    best_path = None
    best_peak = float('inf')
    best_finish = None

    # maps for quick lookup
    iot_map = {iot.id: iot for iot in iot_objs}
    gb_map = {g.id: g for g in gb_objs}

    # try inserting IoT at any position 1..P_len-1 (between waypoints)
    for i_pos in range(1, P_len):
        # compute IoT waypoint w.r.t. predecessor waypoint (we'll assume approach from predecessor)
        # but for candidate building we'll use the generic point_on_line method inside evaluate,
        # so here we insert a placeholder and let evaluate use the waypoint coordinates we compute now.
        # We'll compute i_wp relative to predecessor orig[i_pos-1] position:
        prev_pos = orig[i_pos - 1][2]
        i_wp = point_on_line_at_distance_from_target(prev_pos, iot_map[iot_id].pos, R_I)

        # Now consider options: (A) insert IoT only; (B) insert IoT then GBS after (various positions)
        # We'll attempt inserting IoT-only (no explicit GBS) and IoT+GBS inserted right after IoT (that's sufficient)
        # (paper considers GBS positions after IoT; we try immediate insertion and allow removing duplicates)
        # Candidate 1: IoT only at i_pos
        cand1 = copy.deepcopy(orig)
        cand1.insert(i_pos, ('iot', iot_id, i_wp))
        # clean consecutive GBs in cand1
        cand1 = clean_path_remove_consecutive_gbs(cand1, target)

        # Candidate 2: IoT at i_pos, GBS right after at i_pos+1 (GBS waypoint computed from i_wp)
        g_wp = point_on_line_at_distance_from_target(i_wp, gb_map[gb_id].pos, R_G)
        cand2 = copy.deepcopy(orig)
        cand2.insert(i_pos, ('iot', iot_id, i_wp))
        if dist(i_wp, g_wp) < 3 * R_GBS:
            cand2.insert(i_pos + 1, ('gb', gb_id, g_wp))
        cand2 = clean_path_remove_consecutive_gbs(cand2, target)

        candidates = [cand1, cand2]

        for cand in candidates:
            # build temp UAV list with candidate path for target UAV
            temp_uavs = []
            for u in uav_list:
                if u.id == target_uav_id:
                    u_temp = u.copy()
                    u_temp.path = cand
                    temp_uavs.append(u_temp)
                else:
                    temp_uavs.append(u.copy())
            # simulate
            V_D_glob, V_U_glob, _, per_uav_finish_times, feas = evaluate_all_uavs(temp_uavs, iot_objs, gb_objs)
            # must be feasible for target UAV
            if not feas[target_uav_id]:
                continue
            # compute AoI peak
            _, peak, _, _ = compute_aoi_from_VU(V_U_glob, iot_objs)
            peak_val = float('inf') if peak is None else peak
            if peak_val < best_peak:
                best_peak = peak_val
                best_path = cand
                best_finish = per_uav_finish_times[target_uav_id]

    if best_path is None:
        return None, float('inf'), None
    return best_path, best_peak, best_finish


def clean_path_remove_consecutive_gbs(path, uav):
    """Remove consecutive GB waypoints; ensure start and end are present"""
    cleaned = []
    for entry in path:
        if cleaned and cleaned[-1][0] == 'gb' and entry[0] == 'gb':
            # skip duplicate GBS entries
            continue
        cleaned.append(entry)
    # ensure start and end are correct
    if cleaned[0][0] != 'start':
        cleaned.insert(0, ('start', None, uav.start))
    if cleaned[-1][0] != 'end':
        cleaned.append(('end', None, uav.end))
    return cleaned


# ----------------------------
# Main Greedy Algorithm (Algorithm 3)
# ----------------------------
def greedy_multi_uav_with_ranges(uav_list, iot_objs, gb_objs, seed=None, R_I = R_IOT, R_G = R_GBS):
    if seed is not None:
        random.seed(seed);
        np.random.seed(seed)

    unvisited = {iot.id: iot for iot in iot_objs}
    Lu_cur = {u.id: u.path[-2][2] if len(u.path) >= 2 else u.start for u in uav_list}
    Tcur = {u.id: 0 for u in uav_list}

    while unvisited:
        # each UAV proposes the IoT it can deliver earliest
        candidates = []
        for u in uav_list:
            items = list(unvisited.values())
            if not items:
                continue
            best = earliest_iot_delivery_for_uav(u, Lu_cur[u.id], Tcur[u.id], [it.id for it in items], iot_objs,
                                                 gb_objs)
            iot_id, gb_id, up_time, cap_time, i_wp, g_wp = best
            if iot_id is not None:
                candidates.append((up_time, u.id, iot_id, gb_id, cap_time, i_wp, g_wp))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0])
        up_time, sel_uav_id, sel_iot_id, sel_gb_id, sel_cap_time, sel_i_wp, sel_g_wp = candidates[0]

        # try all insertion positions for selected UAV and select best by global peak AoI
        best_path, best_peak, best_finish = try_insertion_global(uav_list, sel_uav_id, sel_iot_id, sel_gb_id, iot_objs,
                                                                 gb_objs)
        sel_uav = next(u for u in uav_list if u.id == sel_uav_id)

        if best_path is None:
            # fallback: append IoT and its nearest GBS (as boundary points) before the end if feasible
            fallback = copy.deepcopy(sel_uav.path[:-1])
            # compute waypoint relative to last before end
            prev_pos = fallback[-1][2]
            i_wp = point_on_line_at_distance_from_target(prev_pos, unvisited[sel_iot_id].pos, R_I)
            nearest_g = min(gb_objs, key=lambda g: dist(i_wp, g.pos))
            g_wp = point_on_line_at_distance_from_target(i_wp, nearest_g.pos, R_G)
            fallback.append(('iot', sel_iot_id, i_wp))
            fallback.append(('gb', nearest_g.id, g_wp))
            fallback.append(('end', None, sel_uav.end))
            # test feasibility
            temp_uavs = []
            for u in uav_list:
                if u.id == sel_uav_id:
                    u_temp = u.copy();
                    u_temp.path = fallback;
                    temp_uavs.append(u_temp)
                else:
                    ####dilna change
                    temp_uavs.append(u.copy())
            _, _, _, per_uav_finish_times, feas = evaluate_all_uavs(temp_uavs, iot_objs, gb_objs)
            if feas[sel_uav_id]:
                sel_uav.path = fallback
                Lu_cur[sel_uav_id] = sel_uav.path[-2][2]
                Tcur[sel_uav_id] = per_uav_finish_times[sel_uav_id]
                if sel_iot_id in unvisited:
                    del unvisited[sel_iot_id]
            else:
                # cannot deliver by selected UAV -> remove iot from unvisited to avoid infinite loop (could be delivered by other UAVs earlier)
                # (this is conservative; a better approach would re-attempt with other UAVs)
                if sel_iot_id in unvisited:
                    del unvisited[sel_iot_id]
                continue
        else:
            # accept best_path
            sel_uav.path = best_path
            Lu_cur[sel_uav_id] = sel_uav.path[-2][2]
            Tcur[sel_uav_id] = best_finish
            if sel_iot_id in unvisited:
                del unvisited[sel_iot_id]

    # finalize: ensure every UAV uploads its pending captures by adding a GBS if needed
    for u in uav_list:
        V_D, V_U, plen, tfin, feas = evaluate_uav_path(u, iot_objs, gb_objs)
        pending = [iid for iid in V_D if V_D[iid] is not None and V_U[iid] is None]
        if pending:
            nearest_g = min(gb_objs, key=lambda g: dist(u.path[-2][2], g.pos))
            g_wp = point_on_line_at_distance_from_target(u.path[-2][2], nearest_g.pos, R_G)
            # insert before end
            u.path = u.path[:-1] + [('gb', nearest_g.id, g_wp)] + [u.path[-1]]

    return uav_list


# ----------------------------
# Scenario generation + driver
# ----------------------------
def build_random_scenario(num_iot=NUM_IOT, num_gbs=NUM_GBS, num_uavs=NUM_UAV, area=AREA_SIZE, seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    # GBS locations
    gb_list = []
    for gid in range(1, num_gbs + 1):
        x = random.uniform(0.1 * area, 0.9 * area)
        y = random.uniform(0.1 * area, 0.9 * area)
        gb_list.append(GBS(gid, x, y))
    # IoT locations + generation times
    iot_list = []
    for iid in range(1, num_iot + 1):
        x = random.uniform(0, area)
        y = random.uniform(0, area)
        tgen = random.randint(0, 20)
        iot_list.append(IoT(iid, x, y, tgen))
    # UAV start/end points (cycle over corners)
    starts = [(0.05 * area, 0.05 * area), (0.95 * area, 0.05 * area), (0.05 * area, 0.95 * area),
              (0.95 * area, 0.95 * area)]
    ends = [(0.05 * area, 0.95 * area), (0.95 * area, 0.95 * area), (0.95 * area, 0.05 * area),
            (0.05 * area, 0.05 * area)]
    uav_list = []
    for uid in range(1, num_uavs + 1):
        s = starts[(uid - 1) % len(starts)]
        e = ends[(uid - 1) % len(ends)]
        uav_list.append(UAV(uid, s, e, UAV_SPEED, T_MAX))
    return iot_list, gb_list, uav_list


# ----------------------------
# Plotting helpers
# ----------------------------
def plot_solution(iot_list, gb_list, uav_list, title="Greedy multi-UAV with ranges"):
    plt.figure(figsize=(8, 8))
    ixs = [iot.pos[0] for iot in iot_list];
    iys = [iot.pos[1] for iot in iot_list]
    plt.scatter(ixs, iys, marker='o', label='IoT')
    for iot in iot_list:
        plt.text(iot.pos[0] + 0.4, iot.pos[1] + 0.4, f"I{iot.id}(t={iot.t_gen})", fontsize=8)
    gxs = [g.pos[0] for g in gb_list];
    gys = [g.pos[1] for g in gb_list]
    plt.scatter(gxs, gys, marker='s', label='GBS')
    for g in gb_list:
        plt.text(g.pos[0] + 0.4, g.pos[1] + 0.4, f"G{g.id}", fontsize=9)
    colors = ['green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    for u in uav_list:
        path_coords = [np.array(w[2]) for w in u.path]
        px = [p[0] for p in path_coords];
        py = [p[1] for p in path_coords]
        plt.plot(px, py, '--x', label=f"UAV{u.id}", color=colors[(u.id - 1) % len(colors)])
        plt.text(u.start[0], u.start[1], f"U{u.id}_S", fontsize=9)
        plt.text(u.end[0], u.end[1], f"U{u.id}_F", fontsize=9)
    plt.title(title);
    plt.legend();
    plt.grid(True);
    plt.xlim(0, AREA_SIZE);
    plt.ylim(0, AREA_SIZE)
    plt.show()


# ----------------------------
# Run example
# ----------------------------
import datetime
import csv
import time

# --------------------------------------------
# Multi-seed experiment driver
# --------------------------------------------
def run_multi_seed_experiment(
    seeds_list,
    num_iot=NUM_IOT,
    num_gbs=NUM_GBS,
    num_uavs=NUM_UAV,
    area=AREA_SIZE
):
    results = []

    for s in seeds_list:
        print("\n==============================")
        print(f" Running Experiment - Seed {s}")
        print("==============================")

        # Build scenario
        iot_objs, gb_objs, uav_list = build_random_scenario(
            num_iot=num_iot, num_gbs=num_gbs, num_uavs=num_uavs,
            area=area, seed=s
        )

        # Greedy solution
        t0 = time.time()
        uavs_res = greedy_multi_uav_with_ranges(uav_list, iot_objs, gb_objs, seed=s)
        elapsed = time.time() - t0

        # Evaluate performance
        _, V_U_glob, per_uav_lengths, _, _ = evaluate_all_uavs(
            uavs_res, iot_objs, gb_objs
        )

        aoi_dict, peak, avg, coverage = compute_aoi_from_VU(V_U_glob, iot_objs)

        # --------------------------
        # Console print (requested)
        # --------------------------
        print("\nIndividual AoIs (None means not collected):")
        t_gen_map = {}
        for iid in sorted(aoi_dict.keys()):
            t_gen = next(i.t_gen for i in iot_objs if i.id == iid)
            t_gen_map[iid] = t_gen
            print(f" IoT {iid}: AoI = {aoi_dict[iid]} (t_gen={t_gen})")

        # --------------------------
        # Plot network result
        # --------------------------
        #plot_solution(iot_objs, gb_objs, uavs_res,
                      #title=f"Greedy multi-UAV Solution | Seed {s}")

        # Metrics
        unserved = sum(1 for v in aoi_dict.values() if v is None)
        total_dist = sum(per_uav_lengths.values())

        results.append({
            "seed": s,
            "peak_aoi": peak,
            "avg_aoi": avg,
            "coverage": coverage,
            "total_dist": total_dist,
            "unserved": unserved,
            "runtime_sec": elapsed,
            "aoi_dict": aoi_dict,
            "t_gen_map": t_gen_map
        })

        print(f"[Seed {s}] Peak={peak}, Avg={avg:.2f}, Cov={coverage:.2f}, "
              f"Dist={total_dist:.1f}, Unserved={unserved}")

    return results


# --------------------------------------------
# Statistics over all seeds
# --------------------------------------------
def compute_final_stats(results):
    def avg(field):
        vals = [r[field] for r in results if r[field] is not None]
        return sum(vals) / len(vals) if vals else None

    final = {
        "avg_peak_aoi": avg("peak_aoi"),
        "avg_avg_aoi": avg("avg_aoi"),
        "avg_coverage": avg("coverage"),
        "avg_total_dist": avg("total_dist"),
        "avg_unserved": avg("unserved"),
        "avg_runtime": avg("runtime_sec"),
    }
    return final


# --------------------------------------------
# Save results to CSV with detailed AOI logs
# --------------------------------------------
def save_results_csv(results, final_stats, num_iot, num_uavs, num_gbs):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"multi_seed_results_{timestamp}.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # Top: scenario general info
        writer.writerow(["Scenario Information"])
        writer.writerow(["Number of IoT Nodes", num_iot])
        writer.writerow(["Number of UAVs", num_uavs])
        writer.writerow(["Number of Ground BS", num_gbs])
        writer.writerow([])

        # Per-seed performance header
        writer.writerow([
            "Seed", "Peak AOI", "Avg AOI", "Coverage",
            "Total Distance", "Unserved IoT", "Runtime (s)"
        ])

        # Write each seed block
        for r in results:
            writer.writerow([
                r["seed"], r["peak_aoi"], r["avg_aoi"], r["coverage"],
                r["total_dist"], r["unserved"], r["runtime_sec"]
            ])

            # IoT-level AoI detail
            writer.writerow(["Individual AoIs for Seed", r["seed"]])
            for iid in sorted(r["aoi_dict"].keys()):
                writer.writerow([
                    f"IoT {iid}",
                    r["aoi_dict"][iid],
                    f"t_gen={r['t_gen_map'][iid]}"
                ])
            writer.writerow([])

        # Final aggregated metrics
        writer.writerow(["Final Averages"])
        for k, v in final_stats.items():
            writer.writerow([k, v])

    print(f"\n📁 Detailed results saved to: {filename}")


# --------------------------------------------
# Run Experiment (MAIN)
# --------------------------------------------
if __name__ == "__main__":
    seeds = [42,50,55,60,65,85,90,105,49,110]  # Edit if needed

    print("\n===== Running Multi-Seed Experiment =====")
    results = run_multi_seed_experiment(seeds)

    print("\n===== Aggregated Stats =====")
    final_stats = compute_final_stats(results)
    for k, v in final_stats.items():
        print(f"{k}: {v}")

    save_results_csv(results, final_stats,
                     num_iot=NUM_IOT, num_uavs=NUM_UAV, num_gbs=NUM_GBS)

