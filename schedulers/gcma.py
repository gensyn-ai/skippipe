from .graph import *
from .tsp import tsp_dp
from .bipartite_matching import *
import random
import numpy as np
import itertools
def compute_pipeline_parallel_cost(g,candidate_partition):
    cm = [[float("inf") for _ in range(len(candidate_partition))] for _ in range(len(candidate_partition))]
    for idx0,p0 in enumerate(candidate_partition):
        for idx1,p1 in enumerate(candidate_partition):
            if idx1 <= idx0:
                continue
            tmpcm = make_bipartite_graph(g,p0,p1)
            b, _ = bipartite_matching(tmpcm)
            cm[idx0][idx1] = b
            cm[idx1][idx0] = b
    tmpg = Graph(0)
    tmpg.add_cost_matrix(cm)
    # print(cm)
    return tsp_dp(tmpg,start_idx=0)   
            

def compute_data_parallel_cost(g,candidate_partitions):
    score = 0
    for p in candidate_partitions:
        for x in p:
            for y in p:
                score = max(g._cost_matrix[x][y],score)
    return score

def GCMA(g: Graph, population_size=100, trails=2000, mode="Default", partition_sizes = [], factor = 2):
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
            for partition_idx in cur_offspring:
                loc_partition_sizes[partition_idx] += 1

            gain = np.zeros(shape=(num_devices, way))
            for v_idx, partition_idx in enumerate(cur_offspring):
                if locked_v_idx[v_idx] == 0:
                    gain[v_idx][partition_idx] = np.inf
                    for target_idx, target_partition_idx in enumerate(cur_offspring):
                        partial_pipeline_parallel_cost = g._cost_matrix[v_idx][target_idx] * factor
                        if partition_idx != target_partition_idx:
                            gain[v_idx][target_partition_idx] += partial_pipeline_parallel_cost / \
                                loc_partition_sizes[target_partition_idx]
                        elif v_idx != target_idx:
                            if gain[v_idx][target_partition_idx] > partial_pipeline_parallel_cost:
                                gain[v_idx][target_partition_idx] = partial_pipeline_parallel_cost

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

        def calculate_gain_baseline(cur_offspring=None, locked_v_idx=None):
            gain = np.zeros(shape=(num_devices, way))
            for v_idx, partition_idx in enumerate(cur_offspring):
                if locked_v_idx[v_idx] == 0:
                    for target_idx, target_partition_idx in enumerate(cur_offspring):
                        partial_pipeline_parallel_cost = g._cost_matrix[v_idx][target_idx]
                        partial_data_parallel_cost = g._cost_matrix[v_idx][target_idx]*factor
                        if v_idx != target_idx:
                            gain[v_idx][target_partition_idx] += partial_pipeline_parallel_cost
                            gain[v_idx][target_partition_idx] -= partial_data_parallel_cost

            G_i_trace = [[None, None] for i in range(way)]
            G_i = np.full(shape=(way), fill_value=-np.inf)
            G_ij = np.full(shape=(way, way), fill_value=-np.inf)
            for v_idx, partition_idx in enumerate(cur_offspring):
                if locked_v_idx[v_idx] == 0:
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
                if mode == "default":
                    G_ij, G_i, G_i_trace = calculate_gain_default(
                        cur_offspring, locked_v_idx)
                else:
                    G_ij, G_i, G_i_trace = calculate_gain_baseline(
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
                    if mode == "default":
                        G_ij, G_i, G_i_trace = calculate_gain_default(
                            cur_offspring, locked_v_idx)
                    else:
                        G_ij, G_i, G_i_trace = calculate_gain_baseline(
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

        
        data_parallel_cost = compute_data_parallel_cost(g,tmp)
        pipeline_parallel_cost, _ = compute_pipeline_parallel_cost(g,tmp)
        candidate_scores.append(data_parallel_cost +
                                2 * pipeline_parallel_cost)
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
        offspring_data_parallel_cost = compute_data_parallel_cost(g,offspring)
        offspring_pipeline_parallel_cost, offspring_pipeline_parallel_path = compute_pipeline_parallel_cost(g,
            offspring)
        offspring_score = offspring_data_parallel_cost + \
            2 * offspring_pipeline_parallel_cost
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
    
            
            