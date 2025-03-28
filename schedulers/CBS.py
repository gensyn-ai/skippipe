from typing import Dict, List, Union, Tuple, Any
import numpy as np
import heapq
from dataclasses import dataclass, field
from .graph import *
from .bipartite_matching import *
from .a_star_modified import *
import asyncio
import time
from random import random
from .com_sym import *
from joblib import Parallel, delayed
from copy import deepcopy
@dataclass(order=True)
class CBS_item:
    dist: int
    conflict_number: int
    solution: List=field(compare=False)
    conflicts: List[Conflict]=field(compare=False)
    visitable: Dict[int,List[int]]=field(compare=False)
    visitable_stages: Any=field(compare=False)
    uniq_str: str=field(compare=False)
    speeds: List=field(compare=False)
    conflict_3: bool
@dataclass
class CountDC:
    ag: int
    nd: int
    enter_time: float
    exit_time: float
    prv_nd: int
    nxt_nd: int

@dataclass
class VisitDC:
    ag_idx: int
    nd: int
    enter_time: float
    exit_time: float
    prv_nd: int
    forward: bool
    
def visualise_type_3_conflicts(sol):
    for ag_sol in sol.solution:
                                            
        print(ag_sol[0],ag_sol[2])
                                            
        for nd in ag_sol[1]:
            print(nd)
    for c in sol.conflicts:
        if c.type == 3:
            print(c.agidx,c.ndix,c.tmstart,c.tmend)
    return
import itertools
def speed_ranking(speeds, el):
    for idx, s in enumerate(speeds):
        if s[0] == el:
            return idx
def CBS(g:Graph, agents: list[Agent], heuristic, partitions, constraints = [False, True, True], path_l = 4, memory = 3, mb_per_stage_max = 9,delta = 1,limit_TC1 = False):
    h: List[CBS_item] = []
    heapq.heapify(h)
    print(mb_per_stage_max,len(agents))
    visitable: Dict[int,List[int]] = np.full((len(agents),len(g.nodes)),1)
    visitable_stages = np.full((len(agents),len(partitions)),1)
    
    
    conflicts: List[Conflict] = []
    visited: Dict[str,bool] = dict()
   
    curr = time.time()
    
    results = Parallel(n_jobs=len(agents))(delayed(a_star_modified)(g,a.start_idx, heuristic, a.idx, a.dt, conflicts, visitable, visitable_stages, path_l) for a in agents)
    print(time.time()-curr)
    
    cost = 0
    speeds = []
    for v in results:
                if v == None:
                    cost = float("inf")
                    continue
                speeds.append((v[2],v[0]))
                cost = max(v[0],cost)
 
    speeds.sort(key=lambda el: el[1],reverse= True)
    
    heapq.heappush(h,CBS_item(cost,0,results,conflicts, visitable, visitable_stages,"",speeds,False))
    count_per_node: Dict[int,List[Visits]] = dict()
    count_per_partitions: Dict[int,int] = dict()
    for k,v in g.nodes.items():
        count_per_node[k] = []
    visits_per_node: Dict[int,List[VisitDC]] = dict()
    for k,v in g.nodes.items():
        visits_per_node[k] = []
    for v in range(len(partitions)):
        count_per_partitions[v] = []
    solutions = []
    last_value = 0
    curr_smallest = float("inf")
    best_sol = None
    check_1 = False
    check_2 = False
    check_3 = False
    solutions_considered = dict()
    final_solutions = []
    min_conflicts_2 = float("inf")
    first_solution = None
    while len(h) > 0:

        # clear out what we do not need anymore:
        for k,v in g.nodes.items():
            count_per_node[k].clear()
        for k,v in g.nodes.items():
            visits_per_node[k].clear()
        for v in range(len(partitions)):
            count_per_partitions[v].clear()
        
        
        if (len(solutions) >= 16 and not check_1) or (not check_1 and len(h) == 1 and len(solutions) > 1):
            print(len(solutions),"viable solutions found")
            h = []
            solutions_considered.clear()
            heapq.heapify(h)
            for s in solutions:
                
                heapq.heappush(h,s)
            if check_1:
                check_2
            
            solutions = []
            check_1 = True
            
        sol = heapq.heappop(h)
        # print(sol.dist, len(h),curr_smallest)
        
        if not check_1:
            # if len(sol.conflicts) > 1 and sol.visitable_stages.data.tobytes() in visited:
            #     continue
            visited[sol.visitable_stages.data.tobytes()] = True
        else:
            
            if not sol.conflict_3 and np.sum(sol.visitable) < sol.visitable.shape[1]*sol.visitable.shape[0] and (sol.visitable_stages.data.tobytes()+sol.visitable.data.tobytes() in visited or sol.visitable.tobytes() in visited):
                # print("duplicate")
                continue
            visited[sol.visitable_stages.data.tobytes()+sol.visitable.data.tobytes()] = True
            visited[sol.visitable.data.tobytes()] = True


        # check for conflicts
        # Check times on nodes:
        for ag_sol in sol.solution:
            first_node = ag_sol[1][0][0]
            prv_nd = None
            prv_tm = 0
            count = 0
            # print(len(ag_sol[1]))
            el_history = []
            for nd in ag_sol[1]:
                
                if nd[0] == first_node and count == 1:
                    break
                elif nd[0] == first_node:
                    count += 1
                count_per_partitions[g.nodes[nd[0]].properties["partition"]].append((ag_sol[2],nd[1]))
                
                _tmp = CountDC(ag_sol[2], nd[0], nd[1],  ag_sol[0], prv_nd, None)
                count_per_node[nd[0]].append(_tmp)
                if len(el_history) > 0:
                    el_history[-1].nxt_nd = nd[0]
                el_history.append(_tmp)
                
                prv_nd = nd[0]
                prv_tm = nd[1]
                
            el_history[-1].nxt_nd = first_node
            prv_nd = None
            count = 0
            for nd in ag_sol[1]:
                
                if nd[0] == first_node:
                    count += 1
                
                visits_per_node[nd[0]].append(VisitDC(ag_sol[2],nd[0],nd[1],nd[2],prv_nd, count == 1))
                
                
                
                prv_nd = nd[0]
                
                
        
        conflicts = []
        heuristic = 0
        impossible = False
        flag = False
        # CONFLICT TYPE 2:
        if constraints[1]:
            # print("CHECKING CONSTRAINT 1")
            for k in range(len(partitions)):
                
                if k == 0:
                    continue
                if len(count_per_partitions[k]) > mb_per_stage_max:
                    if flag:
                        # heuristic += 0.001 * (len(count_per_partitions[k]) - mb_per_stage_max)
                        heuristic += 0.0001
                        continue
                    # we have found a partition with more than the max nodes per stage
                    flag = True
                    # print(k)
                    count_per_partitions[k].sort(key = lambda el: speed_ranking(sol.speeds,el[0]),reverse = True)
                    
                    # print(count_per_partitions[k])
                    # 
                    # exit()
                    ttl_count = len(count_per_partitions[k])
                    # print(k,count_per_partitions[k])
                    exclude_agents = []
                    for ag in agents:
                        if np.sum(sol.visitable_stages[ag.idx]) == path_l:
                            exclude_agents.append(ag.idx)
                        
                    count_per_partitions[k] = [ t for t in count_per_partitions[k] if t[0] not in exclude_agents]
                    
                    if len(exclude_agents) > mb_per_stage_max:
                        impossible = True
                        break
                    
                    for comb in itertools.combinations(count_per_partitions[k],1 if delta > 1 else ttl_count - mb_per_stage_max):
                        # print(comb)
                        if len(conflicts) > (1 + 3 / (3/delta)):
                            break
                        tmp = []
                        for c in comb:
                            
                            tmp.append(Conflict(c[0],k,-1000,float("inf"),2))
                        
                        # print(comb,"cant visit")
                        # exit()
                        conflicts.append(tmp)
                    # exit()
                    
        if not flag and not check_1:
            
            print("solution for 1 found")
            # print(len(solutions),32//(delta**2))
            sol.visitable_stages[:,:] = 0
            sol.visitable_stages[:,0] = 1
            
            for k in range(len(partitions)):
                if k == 0:
                    continue
                for t in count_per_partitions[k]:
                    # print(k,t)
                    sol.visitable_stages[t[0]][k] = 1
            if sol.visitable_stages.data.tobytes() in solutions_considered:
                continue
            for nd in g.nodes.values():
                partition_node = nd.properties["partition"]
                for ag in agents:
                    if sol.visitable_stages[ag.idx][partition_node] == 0:
                        sol.visitable[ag.idx][nd.idx] = 0
            # print(sol.visitable)
            # print(sol.visitable_stages)
            solutions_considered[sol.visitable_stages.data.tobytes()] = True
            solutions.append(sol)
            continue
        
        
        if check_1 and not flag:
            # type 1 constraints
            for k in range(len(g.nodes)):
                # continue
                
                if g.nodes[k].properties["partition"] == 0:
                    continue
                if len(count_per_node[k]) > memory:
                    if flag == True:
                        heuristic += 0.0001 * (len(count_per_node[k]) - memory)
                        # heuristic += 0.0001
                        continue 
                    # a node has memory exceeded
                    flag = True
                    # print(k)
                    this_partition = g.nodes[k].properties["partition"]
                    this_partition = partitions[this_partition]
                    # print(sol.speeds)
                    count_per_node[k].sort(key = lambda el: speed_ranking(sol.speeds,el.ag),reverse=True)
                    
                    ttl_agents = len(count_per_node[k])
                    exclude_agents = []
                    for ag in count_per_node[k]:
                        ag = ag.ag
                        summ = 0
                        for nd in this_partition:
                            if nd == k:
                                continue
                            summ += sol.visitable[ag][nd]
                        if summ == 0:
                            exclude_agents.append(ag)
                        
                    
                    if len(exclude_agents) > memory:
                        impossible = True
    
                        break
                    count_per_node[k] = [ t for t in count_per_node[k] if t.ag not in exclude_agents]
                    # [:ttl_count-(memory-1)], ttl_agents - memory)
                    for comb in itertools.combinations(count_per_node[k], 1 if delta > 1 else ttl_agents - memory):
                        if len(conflicts) >  (2):
                            break
                        tmp = []
                        for c in comb:
                            tmp.append(Conflict(c.ag,k,-1000,float("inf"),1))
                        
                        conflicts.append(tmp)
                    
        if impossible:
            continue
        
        
        if not flag and not limit_TC1:
            check_2 = True
        if flag:
            if curr_smallest < 1000 and len(h) > 500:
                continue
        

        conflict_3 = False
        if check_2 and constraints[2] and not flag:
            # print("CHECKING TYPE 3",len(h), sol.dist)
            
            checked = []
            for ag_idx_consider,ag in enumerate(sol.speeds):
                
                if len(conflicts) > 0:
                    break
                for ag_idx_consider2,ag2 in enumerate(sol.speeds):
                    if ag_idx_consider >= ag_idx_consider2:
                        continue
                    if len(conflicts) > 0:
                        break
                    for k in range(len(g.nodes)):
                        for idx, visit in enumerate(visits_per_node[k]):
                            if len(conflicts) > 0:
                                break

                            if visit.ag_idx != ag[0]:
                                continue
                            for idx2, visit2 in enumerate(visits_per_node[k]):
                                if len(conflicts) > 0:
                                    break
                                if visit.nd != visit2.nd:
                                    continue
                                if visit.ag_idx == visit2.ag_idx:
                                    continue
                                if visit2.ag_idx != ag2[0]:
                                    continue
                                
                                if (visit.enter_time <= visit2.enter_time  and visit.exit_time > visit2.enter_time ) or (visit2.enter_time <= visit.enter_time  and visit2.exit_time > visit.enter_time ):
                                    if not visit2.forward and not visit.forward:
                                        continue
                                    flag = True
                                    checked.append(visit.ag_idx)
                                    checked.append(visit2.ag_idx)
                                    
                                    conflict_3 = True
                                    
                                    if not visit.forward or  not visit2.forward:
                                        
                                        # prefer faster one                                        
                                        if visit.enter_time < visit2.enter_time:
                                            conflicts.append([Conflict(visit2.ag_idx,k,visit.enter_time,visit.exit_time,3)]) 
                                        else:
                                            conflicts.append([Conflict(visit.ag_idx,k,visit2.enter_time,visit2.exit_time,3)]) 

                                    else:
                                        
                                        if ag_idx_consider > len(agents)*1/4: # rule of thumb

                                            conflicts.append([Conflict(visit.ag_idx,k,visit2.enter_time,visit2.exit_time,3)])  
                                        # print("adding conflict",visit2[0],k,visit[1],visit[2])
                                        conflicts.append([Conflict(visit2.ag_idx,k,visit.enter_time,visit.exit_time,3)])
                                
                                    break
        

        if not flag:
            if not constraints[2]:
                return sol
            final_solutions.append(sol)
            # visualise_type_3_conflicts(sol)
            print("SOME SOLUTION FOUND...",sol.dist)
            if len(final_solutions) >= 8//(delta**2): # check if we do find the best solution
                if best_sol != None:
                    final_solutions.append(best_sol)
                return final_solutions
            continue
        for c in conflicts:
            if len(c) == 0:
                continue
            tmpvisitable_stages = deepcopy(sol.visitable_stages.copy())
            tmpvisitable = deepcopy(sol.visitable.copy())
            for conf in c:
                if conf.type == 2:
                    # print(conf.agidx,"CANNOT VISIT",conf.ndix)
                    tmpvisitable_stages[conf.agidx][conf.ndix] = 0
                    # print(tmpvisitable_stages[conf.agidx])
                    for nd_p in partitions[conf.ndix]:
                        tmpvisitable[conf.agidx][nd_p] = 0
                elif conf.type == 1:
                    tmpvisitable[conf.agidx][conf.ndix] = 0
                   
                    count = 0
                    for nd_p in partitions[g.nodes[conf.ndix].properties["partition"]]:
                        count += tmpvisitable[conf.agidx][nd_p]
                    if count == 0:
                        tmpvisitable_stages[conf.agidx][g.nodes[conf.ndix].properties["partition"]] = 0
                        
                        
                        
                        
                    
            

            
            comb = c + sol.conflicts.copy()
            
            curr = time.time()
            results = Parallel(n_jobs=len(agents))(delayed(a_star_modified)(g,a.start_idx, heuristic, a.idx, a.dt, comb, tmpvisitable, tmpvisitable_stages, path_l) for a in agents)
            # print(time.time()-curr)

            cost = 0
            speeds = []
            for v in results:
                if v == None:
                    cost = float("inf")
                    
                    continue
                speeds.append((v[2],v[0]))
                cost = max(v[0],cost)
            if cost > 100000:
                continue
            
            if conflict_3:
                min_conflicts_2 = min(sol.dist,min_conflicts_2)
                nds = {}
                for idx in range(len(g.nodes)):
                    nds[idx] = ComNode(g._wm[idx],idx,3)
                for ag in results:
                    # for mb_c in range(3):
                        for d in range(2):
                            mb_c = 0
                            path = {}
                            prv = None
                            for t in ag[1]:
                                
                                t = t[0]
                                if prv == None:
                                    prv = t
                                    continue
                                if t == agents[ag[2]].start_idx:
                                    break
                                path[prv] = t
                                prv = t
                            
                            tmp = MB(d*100 + ag[2],path,agents[ag[2]].start_idx)
                            tmp.tm = d*2.5 + mb_c*0.2 + agents[ag[2]].dt
                            tmp.tm_receive = 10000
                        
                            
                            nds[agents[ag[2]].start_idx].receive(tmp)
                
                tmp_cost = run_simulation(nds,partitions,g._cost_matrix)
                if tmp_cost < curr_smallest:
                    print("New smallest",tmp_cost)
                    best_sol = CBS_item(tmp_cost,0,results,comb, tmpvisitable, tmpvisitable_stages, "",speeds, conflict_3 )
                
               
                curr_smallest = min(curr_smallest,tmp_cost)
                if first_solution == None:
                    first_solution = curr_smallest
                
                if curr_smallest < 0.92 * first_solution: # realistically you will seldom achieve greater speed up...
                    # TODO: find a better way to prune solutions to achieve lower scores
                    # once achieved 0.85 after an hour of running
                    # Pruning techniques would be much appreciated
                    return [best_sol]
                if tmp_cost > curr_smallest + 5: # prevent search in too high solutions...
                    continue

                
                    
                

                
            speeds.sort(key=lambda el: el[1],reverse= True)
            # count_col = count_conflicts(g,results,count_per_node) if conflict_3 else 0
            # print(count_col)
            extra_h = np.sum(tmpvisitable)
            if extra_h == tmpvisitable.shape[0]*tmpvisitable.shape[1] or delta > 1:
                extra_h = 0
            heapq.heappush(h,CBS_item(cost + heuristic + extra_h/(100**delta),-len(comb) if limit_TC1 else 0,results,comb, tmpvisitable, tmpvisitable_stages, "",speeds,conflict_3 ))


    
        for k,v in g.nodes.items():    
            count_per_node[k].clear()
        for k in count_per_partitions.keys():
            count_per_partitions[k].clear()
    
    print("NO SOLUTION")
    
    return [best_sol]