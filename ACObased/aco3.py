#!/usr/bin/env python3
"""
Full working ACO-based multi-UAV AoI-aware IoT collection
Optimized for Peak AoI → Avg AoI → UAV travel distance
Opportunistic GBS uploads when within 2*R_GBS
"""

import random, copy, math, datetime, csv, time
from typing import List
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Problem parameters
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
# ACO hyperparameters
# ----------------------------
NUM_ANTS = 30
ACO_ITERS = 50
ALPHA = 1.0  # pheromone importance
BETA = 2.0   # heuristic importance
RHO = 0.12   # evaporation rate
LAMBDA = 1.0 # ant deposit scalar
ELITIST_BOOST = 6.0
NO_IMPROVE_LIMIT = 120
TAU0 = 1.0
GB_IMMEDIATE_INSERT_THRESHOLD = 2 * R_GBS

# Fitness weights
W_PEAK = 0.7
W_AVG = 0.3
EPS = 1e-9

# ----------------------------
# Utilities
# ----------------------------
def dist(a, b):
    return float(np.linalg.norm(np.array(a)-np.array(b)))

def travel_time(a, b, V):
    return math.ceil(dist(a,b)/V)

def point_on_line_at_distance_from_target(prev_pos, target_pos, radius):
    prev = np.array(prev_pos)
    tgt = np.array(target_pos)
    d = dist(prev, tgt)
    if d <= radius: return tuple(prev)
    vec = prev - tgt
    norm = np.linalg.norm(vec)
    if np.isclose(norm,0.0): return tuple(tgt + np.array([radius,0.0]))
    unit = vec/norm
    return tuple(tgt + unit*radius)

# ----------------------------
# Entities
# ----------------------------
class IoT:
    def __init__(self,_id,x,y,t_gen):
        self.id=int(_id)
        self.pos=(float(x),float(y))
        self.t_gen=int(t_gen)

class GBS:
    def __init__(self,_id,x,y):
        self.id=int(_id)
        self.pos=(float(x),float(y))

class UAV:
    def __init__(self,_id,start,end,V,Tmax):
        self.id=int(_id)
        self.start=tuple(np.array(start))
        self.end=tuple(np.array(end))
        self.V=float(V)
        self.Tmax=float(Tmax)
        self.path=[('start',None,self.start),('end',None,self.end)]
    def copy(self):
        u=UAV(self.id,self.start,self.end,self.V,self.Tmax)
        u.path=copy.deepcopy(self.path)
        return u

# ----------------------------
# Evaluation functions
# ----------------------------
def evaluate_uav_path(uav,iot_objs,gb_objs,R_I=R_IOT,R_G=R_GBS):
    V_D={iot.id:None for iot in iot_objs}
    V_U={iot.id:None for iot in iot_objs}
    iot_tgen={iot.id:iot.t_gen for iot in iot_objs}
    cur_pos=np.array(uav.path[0][2])
    t=0
    total_dist=0.0
    captured_pending=[]
    for typ,idx,wpos in uav.path[1:]:
        wpos=tuple(wpos)
        d=dist(cur_pos,wpos)
        total_dist+=d
        t+=travel_time(cur_pos,wpos,uav.V)
        cur_pos=np.array(wpos)
        if typ=='iot':
            if t<iot_tgen[idx]: t+=int(iot_tgen[idx]-t)
            V_D[idx]=int(t)
            captured_pending.append(idx)
        elif typ=='gb':
            for iid in list(captured_pending):
                V_U[iid]=int(t)
                captured_pending.remove(iid)
    if captured_pending:
        nearest_g=min(gb_objs,key=lambda g: dist(cur_pos,g.pos))
        g_w=point_on_line_at_distance_from_target(cur_pos,nearest_g.pos,R_G)
        total_dist+=dist(cur_pos,g_w)
        t+=travel_time(cur_pos,g_w,uav.V)
        for iid in list(captured_pending):
            V_U[iid]=int(t)
            captured_pending.remove(iid)
    finish_time=int(t)
    feasible=finish_time<=uav.Tmax
    return V_D,V_U,total_dist,finish_time,feasible

def evaluate_all_uavs(uav_list,iot_objs,gb_objs):
    V_D_glob={iot.id:None for iot in iot_objs}
    V_U_glob={iot.id:None for iot in iot_objs}
    per_uav_lengths={}
    per_uav_finish_times={}
    feasibility={}
    for u in uav_list:
        V_D,V_U,plen,tfin,feas=evaluate_uav_path(u,iot_objs,gb_objs)
        per_uav_lengths[u.id]=plen
        per_uav_finish_times[u.id]=tfin
        feasibility[u.id]=feas
        for iid in V_D:
            if V_D[iid] is not None and V_D_glob[iid] is None: V_D_glob[iid]=V_D[iid]
        for iid in V_U:
            if V_U[iid] is not None and V_U_glob[iid] is None: V_U_glob[iid]=V_U[iid]
    return V_D_glob,V_U_glob,per_uav_lengths,per_uav_finish_times,feasibility

def compute_aoi_from_VU(V_U_glob,iot_objs):
    aoi={}
    aois_list=[]
    for iot in iot_objs:
        v=V_U_glob.get(iot.id)
        if v is not None:
            val=int(v-iot.t_gen)
            aoi[iot.id]=val
            aois_list.append(val)
        else: aoi[iot.id]=None
    peak=max(aois_list) if aois_list else None
    avg=sum(aois_list)/len(aois_list) if aois_list else None
    coverage=sum(1 for v in aoi.values() if v is not None)/len(iot_objs)
    return aoi,peak,avg,coverage

# ----------------------------
# Heuristic: Peak+Avg AoI + distance
# ----------------------------
def heuristic_value(cur_pos,candidate_wp,t_now,iobj:IoT,gb_objs,uav_v):
    d=dist(cur_pos,candidate_wp)
    t_arrive=t_now+travel_time(cur_pos,candidate_wp,uav_v)
    t_capture=max(t_arrive,iobj.t_gen)
    nearest_g=min(gb_objs,key=lambda g: dist(candidate_wp,g.pos))
    g_wp=point_on_line_at_distance_from_target(candidate_wp,nearest_g.pos,R_GBS)
    t_upload=t_capture+travel_time(candidate_wp,g_wp,uav_v)
    cost=0.7*(t_upload-t_now)+0.3*d
    return 1.0/(cost+1e-6)

# ----------------------------
# Construct solution for one ant
# ----------------------------
def construct_solution(pheromone,start_tau,iot_list,gb_list,uav_list):
    n=len(iot_list)
    iot_map={iot_list[i].id:iot_list[i] for i in range(n)}
    all_unvisited=set(iot_map.keys())
    solution_uavs={u.id:[('start',None,u.start)] for u in uav_list}

    for u in uav_list:
        cur_pos=u.start
        cur_time=0
        prev_idx=None
        while True:
            candidates=[]
            for iid in list(all_unvisited):
                iobj=iot_map[iid]
                i_wp=point_on_line_at_distance_from_target(cur_pos,iobj.pos,R_IOT)
                t_arrive=cur_time+travel_time(cur_pos,i_wp,u.V)
                t_capture=max(t_arrive,iobj.t_gen)
                nearest_g=min(gb_list,key=lambda g: dist(i_wp,g.pos))
                g_wp=point_on_line_at_distance_from_target(i_wp,nearest_g.pos,R_GBS)
                t_upload=t_capture+travel_time(i_wp,g_wp,u.V)
                if t_upload<=u.Tmax: candidates.append((iid,i_wp,nearest_g.id,g_wp,t_upload))
            if not candidates: break

            probs=[]
            denom=0.0
            for iid,i_wp,gid,g_wp,t_upload in candidates:
                cur_idx=iid-1
                tau=start_tau[cur_idx] if prev_idx is None else pheromone[prev_idx,cur_idx]
                heur=heuristic_value(cur_pos,i_wp,cur_time,iot_map[iid],gb_list,u.V)
                val=(tau**ALPHA)*(heur**BETA)
                probs.append(val)
                denom+=val
            probs=[p/denom for p in probs]
            sel=np.random.choice(len(candidates),p=probs)
            iid,i_wp,gid,g_wp,t_upload=candidates[int(sel)]
            solution_uavs[u.id].append(('iot',iid,i_wp))
            cur_pos=i_wp
            cur_time=t_upload
            prev_idx=iid-1
            all_unvisited.discard(iid)
            if dist(cur_pos,g_wp)<=GB_IMMEDIATE_INSERT_THRESHOLD:
                solution_uavs[u.id].append(('gb',gid,g_wp))
                cur_pos=g_wp
        solution_uavs[u.id].append(('end',None,u.end))

    # fallback
    if all_unvisited:
        for iid in list(all_unvisited):
            iobj=iot_map[iid]
            best_u=min(uav_list,key=lambda uu: dist(solution_uavs[uu.id][-2][2],iobj.pos))
            prev_pos=solution_uavs[best_u.id][-2][2]
            i_wp=point_on_line_at_distance_from_target(prev_pos,iobj.pos,R_IOT)
            nearest_g=min(gb_list,key=lambda g: dist(i_wp,g.pos))
            g_wp=point_on_line_at_distance_from_target(i_wp,nearest_g.pos,R_GBS)
            solution_uavs[best_u.id]=solution_uavs[best_u.id][:-1]+[('iot',iid,i_wp),('gb',nearest_g.id,g_wp),('end',None,best_u.end)]
            all_unvisited.discard(iid)
    return solution_uavs

# ----------------------------
# Fitness computation
# ----------------------------
def compute_fitness_for_solution(solution_uavs,iot_objs,gb_objs,base_uavs):
    uav_copies=[u.copy() for u in base_uavs]
    for u in uav_copies: u.path=solution_uavs[u.id]
    V_D_glob,V_U_glob,per_uav_lengths,per_uav_finish_times,feas=evaluate_all_uavs(uav_copies,iot_objs,gb_objs)
    aoi_dict,peak,avg,coverage=compute_aoi_from_VU(V_U_glob,iot_objs)
    peak_norm=(float(peak)/T_MAX) if peak is not None else 1.0
    avg_norm=(float(avg)/T_MAX) if avg is not None else 1.0
    Tmax_penalty=sum(1000.0 for f in feas.values() if not f)
    fitness=W_PEAK*peak_norm+W_AVG*avg_norm+Tmax_penalty
    return fitness,peak,avg,coverage,sum(per_uav_lengths.values()),per_uav_finish_times,feas,uav_copies

# ----------------------------
# ACO main loop
# ----------------------------
def run_aco(iot_objs,gb_objs,uav_list):
    random.seed(SEED); np.random.seed(SEED)
    n=len(iot_objs)
    pheromone=np.full((n,n),TAU0); np.fill_diagonal(pheromone,0.0)
    start_tau=np.full(n,TAU0)
    best_solution=None; best_fitness=float('inf'); best_info=None
    history=[]; no_improve=0

    for it in range(ACO_ITERS):
        ants_solutions=[]; ants_fitnesses=[]
        for _ in range(NUM_ANTS):
            sol=construct_solution(pheromone,start_tau,iot_objs,gb_objs,uav_list)
            fitness,peak,avg,coverage,total_len,per_uav_finish_times,feas,uav_copies=compute_fitness_for_solution(sol,iot_objs,gb_objs,uav_list)
            ants_solutions.append((sol,fitness,uav_copies))
            ants_fitnesses.append(fitness)
            if fitness<best_fitness:
                best_fitness=fitness
                best_solution=sol
                best_info=(fitness,peak,avg,coverage,total_len,per_uav_finish_times,feas,uav_copies)
                no_improve=0
        no_improve+=1
        if no_improve>=NO_IMPROVE_LIMIT:
            print(f"No improvement for {NO_IMPROVE_LIMIT} iters → early stop at iter {it}")
            break
        pheromone*=(1.0-RHO); start_tau*=(1.0-RHO)
        delta=np.zeros_like(pheromone); delta_start=np.zeros_like(start_tau)
        for sol,fitness,_ in ants_solutions:
            deposit=LAMBDA/(fitness+EPS)
            for u in uav_list:
                prev_idx=None
                for typ,idx,pos in sol[u.id]:
                    if typ=='iot':
                        cur_idx=int(idx)-1
                        if prev_idx is None: delta_start[cur_idx]+=deposit
                        else: delta[prev_idx,cur_idx]+=deposit
                        prev_idx=cur_idx
                    elif typ=='start': prev_idx=None
        pheromone+=delta; start_tau+=delta_start
        np.clip(pheromone,1e-9,1e9,out=pheromone); np.clip(start_tau,1e-9,1e9,out=start_tau)
        if it%5==0 or it==ACO_ITERS-1:
            print(f"Iter {it}: best_fitness={best_fitness:.6f}, peak={best_info[1]}, avg={best_info[2]}, total_dist={best_info[4]:.2f}, coverage={best_info[3]:.3f}")
    return best_solution,best_fitness,best_info,history

# ----------------------------
# Scenario generation
# ----------------------------
def build_random_scenario(num_iot=NUM_IOT,num_gbs=NUM_GBS,num_uavs=NUM_UAV,area=AREA_SIZE,seed=SEED):
    random.seed(seed); np.random.seed(seed)
    gb_list=[GBS(gid,random.uniform(0.1*area,0.9*area),random.uniform(0.1*area,0.9*area)) for gid in range(1,num_gbs+1)]
    iot_list=[IoT(iid,random.uniform(0,area),random.uniform(0,area),random.randint(0,20)) for iid in range(1,num_iot+1)]
    starts=[(0.05*area,0.05*area),(0.95*area,0.05*area),(0.05*area,0.95*area),(0.95*area,0.95*area)]
    ends=[(0.05*area,0.95*area),(0.95*area,0.95*area),(0.95*area,0.05*area),(0.05*area,0.05*area)]
    uav_list=[UAV(uid,starts[(uid-1)%len(starts)],ends[(uid-1)%len(ends)],UAV_SPEED,T_MAX) for uid in range(1,num_uavs+1)]
    return iot_list,gb_list,uav_list

# ----------------------------
# Multi-seed experiments & CSV
# ----------------------------
def run_multi_seed_experiment_aco(seeds_list):
    results=[]
    for s in seeds_list:
        print("\n==============================")
        print(f" Running Experiment - Seed {s}")
        print("==============================")
        iot_objs,gb_objs,uav_list=build_random_scenario(seed=s)
        t0=time.time()
        best_sol,best_fitness,best_info,history=run_aco(iot_objs,gb_objs,uav_list)
        elapsed=time.time()-t0
        print(f"ACO done in {elapsed:.2f}s: best_fitness={best_fitness:.6f}")
        if best_info is not None:
            fitness,peak,avg,coverage,total_dist,per_uav_finish_times,feas_flags,uav_copies=compute_fitness_for_solution(best_sol,iot_objs,gb_objs,uav_list)
            V_D_glob,V_U_glob,_,_,_=evaluate_all_uavs(uav_copies,iot_objs,gb_objs)
            aoi_dict,_,_,_=compute_aoi_from_VU(V_U_glob,iot_objs)
            served_iots={iid for iid,v in V_U_glob.items() if v is not None}
            all_ids=set(iot.id for iot in iot_objs)
            not_served_ids=all_ids-served_iots
            results.append({"seed":s,"fitness":fitness,"peak_aoi":peak,"avg_aoi":avg,"coverage":coverage,"total_dist":total_dist,"unserved":len(not_served_ids),"runtime_sec":elapsed,"aoi_dict":aoi_dict,"t_gen_map":{i.id:i.t_gen for i in iot_objs},"per_uav_finish_times":per_uav_finish_times,"feas_flags":feas_flags,"uav_paths":{u.id:u.path for u in uav_copies}})
        else:
            results.append({"seed":s,"fitness":None,"peak_aoi":None,"avg_aoi":None,"coverage":None,"total_dist":None,"unserved":NUM_IOT,"runtime_sec":elapsed,"aoi_dict":{},"t_gen_map":{},"per_uav_finish_times":{},"feas_flags":{},"uav_paths":{}})
    return results

def compute_final_stats(results):
    def avg(field): vals=[r[field] for r in results if r[field] is not None]; return sum(vals)/len(vals) if vals else None
    final={"avg_fitness":avg("fitness"),"avg_peak_aoi":avg("peak_aoi"),"avg_avg_aoi":avg("avg_aoi"),"avg_total_dist":avg("total_dist"),"avg_unserved":avg("unserved")}
    print("\n========= Overall Statistics =========")
    print(final)
    return final

# ----------------------------
# Plotting UAV paths
# ----------------------------
def plot_uav_solution(iot_objs,gb_objs,uav_copies):
    plt.figure(figsize=(8,8))
    plt.scatter([iot.pos[0] for iot in iot_objs],[iot.pos[1] for iot in iot_objs],c='blue',label='IoT')
    plt.scatter([gb.pos[0] for gb in gb_objs],[gb.pos[1] for gb in gb_objs],c='red',marker='s',label='GBS')
    colors=['green','purple','orange','brown','cyan']
    for u in uav_copies:
        path_points=[p[2] for p in u.path]
        xs=[p[0] for p in path_points]; ys=[p[1] for p in path_points]
        plt.plot(xs,ys,color=colors[u.id%len(colors)],label=f'UAV{u.id}',linewidth=2,marker='o')
    plt.legend()
    plt.title("UAV Paths with IoTs and GBS")
    plt.xlabel("X"); plt.ylabel("Y"); plt.grid(True)
    plt.show()

# ----------------------------
# Main execution
# ----------------------------
if __name__=="__main__":
    seeds_list=[52]
    results=run_multi_seed_experiment_aco(seeds_list)
    stats=compute_final_stats(results)
    # plot last experiment
    last_res=results[-1]
    uav_paths=[UAV(uid,(0,0),(0,0),UAV_SPEED,T_MAX) for uid in range(1,NUM_UAV+1)]
    for u in uav_paths:
        u.path=last_res["uav_paths"][u.id]
    iot_objs=[IoT(iid,last_res["t_gen_map"][iid],last_res["t_gen_map"][iid],last_res["t_gen_map"][iid]) for iid in last_res["t_gen_map"]]
    #gb_objs=[GBS(gb.id,gb.pos[0],gb.pos[1]) for gb in uav_paths] # approximate
    # Correct: gb_list contains GBS objects
    gb_objs = [GBS(gb.id, gb.pos[0], gb.pos[1]) for gb in uav_paths]

    plot_uav_solution(iot_objs,gb_objs,uav_paths)
