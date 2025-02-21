from .graph import *
from .tsp import tsp_dp
from .bipartite_matching import *
import random
import numpy as np
import itertools

def compute_cost(g, candidate_partition):
    max_cost = 0
    for idx, p in enumerate(candidate_partition):
        if idx == len(candidate_partition)-1:
            continue
        partitions_visited = []
        cost = 0
        prv = None
        for nd in p:
            if g.nodes[nd].properties["partition"] in partitions_visited:
                cost = float("inf")
                break
            partitions_visited.append(g.nodes[nd].properties["partition"])
            if prv == None:
                cost += g._wm[nd]
                prv = nd 
                continue
            cost += g._cost_matrix[nd][prv] + g._wm[nd]
            prv = nd
        max_cost = max(max_cost,cost)
    return max_cost

def compute_cost_partition_swap(g,nd1,nd2,p1,p2,pid1,pid2):
    cost_old_1 = 0
    cost_old_2 = 0
    cost_new_1 = 0
    cost_new_2 = 0
    
    partitions_visited1 = []
    partitions_visited2 = []
    prv1 = None
    prv2 = None
    for nd in p1:
        if g.nodes[nd].properties["partition"] in partitions_visited1:
                cost_old_1 = float("inf")
                continue
        if g.nodes[nd].properties["partition"] in partitions_visited2:
                cost_new_1 = float("inf")
                continue
        
        if nd1 == nd:
            partitions_visited1.append(g.nodes[nd1].properties["partition"])
            partitions_visited2.append(g.nodes[nd2].properties["partition"])
            if prv == None:
                cost_old_1 += g._wm[nd1]
                cost_new_1 += g._wm[nd2]
                prv1 = nd1 
                prv2 = nd2
                continue
            cost_old_1 += g._cost_matrix[nd1][prv1] + g._wm[nd1]
            cost_new_1 += g._cost_matrix[nd2][prv2] + g._wm[nd2]
            prv1 = nd1
            prv2 = nd2
        else:
            partitions_visited1.append(g.nodes[nd].properties["partition"])
            partitions_visited2.append(g.nodes[nd].properties["partition"])
            if prv == None:
                cost_old_1 += g._wm[nd]
                cost_new_1 += g._wm[nd]
                prv1 = nd 
                prv2 = nd
                continue
            cost_old_1 += g._cost_matrix[nd][prv1] + g._wm[nd]
            cost_new_1 += g._cost_matrix[nd][prv2] + g._wm[nd]
            prv1 = nd
            prv2 = nd
    if pid1:
        cost_old_1 = 0
        cost_new_1 = 0
    partitions_visited1 = []
    partitions_visited2 = []
    prv1 = None
    prv2 = None
    for nd in p2:
        if g.nodes[nd].properties["partition"] in partitions_visited1:
                cost_old_2 = float("inf")
                continue
        if g.nodes[nd].properties["partition"] in partitions_visited2:
                cost_new_2 = float("inf")
                continue
        
        if nd2 == nd:
            partitions_visited1.append(g.nodes[nd1].properties["partition"])
            partitions_visited2.append(g.nodes[nd2].properties["partition"])
            if prv == None:
                cost_old_2 += g._wm[nd2]
                cost_new_2 += g._wm[nd1]
                prv1 = nd1 
                prv2 = nd2
                continue
            cost_old_2 += g._cost_matrix[nd2][prv1] + g._wm[nd2]
            cost_new_2 += g._cost_matrix[nd1][prv2] + g._wm[nd1]
            prv1 = nd2
            prv2 = nd1
        else:
            partitions_visited1.append(g.nodes[nd].properties["partition"])
            partitions_visited2.append(g.nodes[nd].properties["partition"])
            if prv == None:
                cost_old_2 += g._wm[nd]
                cost_new_2 += g._wm[nd]
                prv1 = nd 
                prv2 = nd
                continue
            cost_old_2 += g._cost_matrix[nd][prv1] + g._wm[nd]
            cost_new_2 += g._cost_matrix[nd][prv2] + g._wm[nd]
            prv1 = nd
            prv2 = nd
    if pid2:
        cost_old_2 = 0
        cost_new_2 = 0
    return (cost_old_1 + cost_old_2) - (cost_new_1 + cost_new_2)



def GCMA_modified(g: Graph, population_size=100, trails=2000, mode="Default", partition_sizes = [], factor = 2, current_partitions = []):
    # https://dl.acm.org/doi/10.5555/2933718.2933740
    num_devices = len(g.nodes)
    way = len(partition_sizes)
    def five_point_crossover(parent1=None, parent2=None):
        parent1_str = [0] * num_devices
        parent2_str = [0] * num_devices
        part = 0
        szs = partition_sizes[0]
        
        for i in range(num_devices):
            
            if i >= szs:
                
                
                part += 1
                szs += partition_sizes[part]
            parent1_str[parent1[i]] = part
            parent2_str[parent2[i]] = part

        points = list(range(num_devices))
        random.shuffle(points)
        points = points[:5]

        for point in points:
            parent2_str[point] = parent1_str[point]

        loc_partition_sizes = [0] * len(partition_sizes)
        for partition_idx in parent2_str:
            loc_partition_sizes[partition_idx] += 1
        for i in range(num_devices):
            if loc_partition_sizes[parent2_str[i]] > partition_sizes[parent2_str[i]]:
                for j in range(len(partition_sizes)):
                    if loc_partition_sizes[j] < partition_sizes[j]:
                        loc_partition_sizes[j] += 1
                        break
                loc_partition_sizes[parent2_str[i]] -= 1
                parent2_str[i] = j
        return parent2_str

    def cyclic_partitioning(offspring=None):
        def calculate_gain_default(cur_offspring=None, locked_v_idx=None):
            loc_partition_sizes = [0] * way
            tmp_partitions = [[] for _ in range(way)]
            for idx,partition_idx in enumerate(cur_offspring):
                loc_partition_sizes[partition_idx] += 1
                tmp_partitions.append(idx)

            gain = np.zeros(shape=(num_devices, way))
            for v_idx, partition_idx in enumerate(cur_offspring):
                if locked_v_idx[v_idx] == 0:
                    gain[v_idx][partition_idx] = np.inf
                    # TODO MODIFY HERE
                    for target_idx, target_partition_idx in enumerate(cur_offspring):
                        partial_pipeline_parallel_cost = compute_cost_partition_swap(g,v_idx, target_idx, tmp_partitions[partition_idx], tmp_partitions[target_partition_idx], partition_idx == way - 1, target_partition_idx == way - 1)
                        gain[v_idx][target_partition_idx] = min(gain[v_idx][target_partition_idx],partial_pipeline_parallel_cost)
                        

            G_i = np.full(shape=(way), fill_value=np.inf)
            G_i_trace = [[None, None] for i in range(way)]
            for v_idx, partition_idx in enumerate(cur_offspring):
                if locked_v_idx[v_idx] == 0:
                    if gain[v_idx][partition_idx] < G_i[partition_idx]:
                        G_i[partition_idx] = gain[v_idx][partition_idx]
                        G_i_trace[partition_idx][0] = v_idx

            G_i = np.full(shape=(way), fill_value=-np.inf)
            G_ij = np.full(shape=(way, way), fill_value=-np.inf)
            for partition_idx, trace in enumerate(G_i_trace):
                v_idx = trace[0]
                if v_idx != None:
                    for target_partition_idx, target_gain in enumerate(gain[v_idx]):
                        if target_partition_idx != partition_idx:
                            target_gain -= gain[v_idx][partition_idx]
                            if target_gain > G_ij[partition_idx, target_partition_idx]:
                                G_ij[partition_idx,
                                     target_partition_idx] = target_gain
                            if target_gain > G_i[partition_idx]:
                                G_i[partition_idx] = target_gain
                                G_i_trace[partition_idx] = [
                                    v_idx, target_partition_idx]

            return G_ij, G_i, G_i_trace



        def move_cycles(offspring=None):
            sum = [0]
            locked_partition_idx = [0] * way
            locked_v_idx = [0] * num_devices
            offsprings = [offspring]
            for _ in range(way):  # how many cycles
                cur_offspring = offsprings[-1].copy()
                movements = []
                epsilon = []
                tau = []
                G_ij, G_i, G_i_trace = calculate_gain_default(
                        cur_offspring, locked_v_idx)
                
                S0 = Si = np.argmax(G_i)
                for _ in range(num_devices):  # how many movement per cycle
                    v_idx, Pv = G_i_trace[Si]
                    if v_idx == None:
                        v_idx = movements[-1][0]
                        Pv = S0
                    cur_offspring[v_idx] = Pv
                    locked_v_idx[v_idx] = 1
                    locked_partition_idx[Pv] = 1
                    movements.append((v_idx, Si, Pv))
                    epsilon.append(G_i[Si])
                    tau.append(G_ij[Si, S0])
                    Si = Pv
                    if Si == S0:
                        break
                    G_ij, G_i, G_i_trace = calculate_gain_default(
                            cur_offspring, locked_v_idx)
                    

                max_sum = 0
                l = 0
                for i in range(1, len(epsilon)):
                    if np.sum(epsilon[:i]) + tau[i] > max_sum:
                        max_sum = np.sum(epsilon[:i]) + tau[i]
                        l = i

                for i in range(len(epsilon) - 1, l, -1):
                    cur_offspring[movements[i][0]] = movements[i][1]
                cur_offspring[movements[l][0]] = S0
                offsprings.append(cur_offspring)
                sum.append(max_sum)

                if np.sum(locked_partition_idx) == len(locked_partition_idx):
                    break

            max_sum = 0
            m = 0
            for i in range(1, len(sum)):
                if np.sum(sum[:i]) > max_sum:
                    max_sum = np.sum(sum[:i])
                    m = i - 1
            offspring = offsprings[m]

            return offspring

        for _ in range(1):
            offspring = move_cycles(offspring)
        return offspring

    candidate_partitions = []
    candidate_scores = []
    candidate_min_scores = []
    for i in range(population_size):
        cur_nodes = list(g.nodes.keys()).copy()
        # TODO: MODIFY HERE
        random.seed = i
        random.shuffle(cur_nodes)
        candidate_partitions.append(cur_nodes)

    for i, candidate_partition in enumerate(candidate_partitions):
        strt = 0
        part = 0
        tmp = []
        while strt < num_devices:
            tmp.append([candidate_partition[ix] for ix in range(strt,strt + partition_sizes[part])])
            strt += partition_sizes[part]
            part += 1

        
        
        candidate_scores.append(compute_cost(g,tmp))
        candidate_min_scores.append(np.min(candidate_scores))

    for i in range(trails):
        np.random.seed = i
        parent1_idx, parent2_idx = np.random.randint(population_size, size=2)
        ga_offspring_str = five_point_crossover(
            candidate_partitions[parent1_idx], candidate_partitions[parent2_idx])
        offspring_str = cyclic_partitioning(ga_offspring_str)

        offspring = [[] for _ in range(way)]
        for v_idx, partition_idx in enumerate(offspring_str):
            offspring[partition_idx].append(v_idx)
        for p in offspring:
            p.sort(key = lambda el: g.nodes[el].properties["partition"])
        
        
        offspring_score = compute_cost(g,offspring)
        offspring = list(itertools.chain.from_iterable(offspring))

        if offspring_score > max(candidate_scores[parent1_idx], candidate_scores[parent2_idx]):
            candidate_partitions.append(offspring)
            candidate_scores.append(offspring_score)
        else:
            replaced_idx = parent1_idx if candidate_scores[
                parent1_idx] > candidate_scores[parent2_idx] else parent2_idx
            replaced_candidate = candidate_partitions[replaced_idx]
            candidate_partitions[replaced_idx] = offspring
            candidate_partitions.append(replaced_candidate)
            replaced_score = candidate_scores[replaced_idx]
            candidate_scores[replaced_idx] = offspring_score
            candidate_scores.append(replaced_score)
        candidate_min_scores.append(np.min(candidate_scores))

    assert(len(candidate_partitions) == len(candidate_scores))
    assert(len(candidate_min_scores) == len(candidate_scores))
    return candidate_partitions, candidate_scores, candidate_min_scores


def reconstruct_partition(g, lst, partition_sizes):
    partions = []
    tmp = []
    sz = partition_sizes[0]
    part = 0
    print(len(lst))
    for i, nd in enumerate(lst):
        if i >= sz:
            
            partions.append(tmp)
            tmp = []
            part += 1
            sz += partition_sizes[part]
        tmp.append(nd)
    partions.append(tmp)
    cost, path = compute_pipeline_parallel_cost(g,partions)
    i = 0
    # print(partions)
    # print(path)
    arrangements = [] 
    for _ in range(len(partions)):
        arrangements.append([])
    prev = None
    while i < len(path):
        # print(i)
        if prev == None:
            prev = partions[path[i]]
            for nd in prev:
                arrangements[i].append(nd)
            i += 1
            continue
        cm = make_bipartite_graph(g,prev,partions[path[i]])
        _, pairs = bipartite_matching(cm)
        
        # print(pairs)
        for ix,p in enumerate(arrangements[i-1]):
            
            for pr in pairs:
                
                if prev[pr[0]] == p and pr[1] < len(partions[path[i]]):
                    arrangements[i].append(partions[path[i]][pr[1]])
                    
        
        prev = arrangements[i]


        i += 1
    # print(arrangements)
    # exit()
    return arrangements
    
            
            