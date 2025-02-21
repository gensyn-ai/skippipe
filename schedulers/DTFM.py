import random
import numpy as np
from scipy.optimize import linear_sum_assignment
import itertools
partition_size = 3
way = 6
def all_candidate_partitions(nodes=None):
    candidate_partitions = []
    if len(nodes) == partition_size:
        candidate_partitions.append([tuple(nodes)])
    else:
        for cur_partition in itertools.combinations(nodes, partition_size):
            rest_nodes = nodes.copy()
            for node in cur_partition:
                rest_nodes.remove(node)

            rest_partitions = all_candidate_partitions(rest_nodes)
            for rest_partition in rest_partitions:
                candidate_partitions.append([cur_partition])
                candidate_partitions[-1].extend(rest_partition)
    return candidate_partitions


def compute_data_parallel_cost(candidate_partition=None,peer_delay=None,Factor = 2):
    data_parallel_cost = float('-inf')
    for partition in candidate_partition:
        within_partition_cost = [0] * partition_size
        for i in range(partition_size):
            for j in range(partition_size):
                if i != j:
                    within_partition_cost[i] += 2 * (Factor*peer_delay[partition[i], partition[j]])
        if data_parallel_cost < np.max(within_partition_cost):
            data_parallel_cost = np.max(within_partition_cost)
    return data_parallel_cost


def compute_pipeline_parallel_cost(candidate_partition=None,peer_delay=None):
    class open_loop_tsp:
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5388488
        def __init__(self, cost_matrix, start_node):
            self.cost_matrix = cost_matrix
            self.num_nodes = self.cost_matrix.shape[0]
            self.start_node = start_node
            self.dp_table = np.full(
                shape=(self.num_nodes, pow(2, self.num_nodes)), fill_value=np.inf)
            self.trace_table = np.zeros(
                shape=(self.num_nodes, pow(2, self.num_nodes)))

        def convert(self, future_nodes):
            binary_future_nodes = 0
            for future_node in future_nodes:
                binary_future_nodes += pow(2, future_node)
            return binary_future_nodes

        def solve(self, node, future_nodes):
            if len(future_nodes) == 0:
                # closed loop tsp problem: return self.cost_matrix[node][self.start_node]
                # open loop tsp problem: return 0
                return 0

            all_distance = []
            for next_node in future_nodes:
                next_future_nodes = future_nodes.copy()
                next_future_nodes.remove(next_node)
                binary_next_future_nodes = self.convert(next_future_nodes)
                if self.dp_table[next_node][binary_next_future_nodes] == np.inf:
                    all_distance.append(
                        self.cost_matrix[node][next_node] + self.solve(next_node, next_future_nodes))
                else:
                    all_distance.append(
                        self.cost_matrix[node][next_node] + self.dp_table[next_node][binary_next_future_nodes])

            min_distance = min(all_distance)
            next_node = future_nodes[all_distance.index(min_distance)]

            binary_future_nodes = self.convert(future_nodes)
            self.dp_table[node][binary_future_nodes] = min_distance
            self.trace_table[node][binary_future_nodes] = next_node
            return min_distance

        def get_least_cost_route(self):
            future_nodes = list(range(self.num_nodes))
            future_nodes.remove(self.start_node)
            cost = self.solve(self.start_node, future_nodes)

            path = [self.start_node]
            cur_node = self.start_node
            while len(future_nodes) > 0:
                binary_future_nodes = self.convert(future_nodes)
                cur_node = int(self.trace_table[cur_node][binary_future_nodes])
                future_nodes.remove(cur_node)
                path.append(cur_node)
            return cost, path

    def bipartite_matching(candidate_partition_0, candidate_partition_1, peer_delay):
        cost_matrix = np.zeros(shape=(partition_size, partition_size))
        for i in range(partition_size):
            for j in range(partition_size):
                cost_matrix[i, j] = peer_delay[candidate_partition_0[i], candidate_partition_1[j]]

        descending_order = np.argsort(cost_matrix.flatten())[::-1]
        inf_weight = 1e6
        for idx in descending_order:
            cur_max_weight = cost_matrix[idx //
                                         partition_size][idx % partition_size]
            cost_matrix[idx//partition_size][idx % partition_size] = inf_weight
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            if cost_matrix[row_ind, col_ind].sum() >= inf_weight:
                return cur_max_weight, list(zip(row_ind, col_ind))

    cross_partition_cost = np.zeros(shape=(way, way))
    dp_pipeline_parallel_match = [
        [None for _ in range(way)] for _ in range(way)]
    for i in range(way):
        for j in range(i+1, way):
            
            cross_partition_cost[i, j], dp_pipeline_parallel_match[i][j] = bipartite_matching(
                candidate_partition[i], candidate_partition[j],peer_delay=peer_delay)
            cross_partition_cost[j, i] = cross_partition_cost[i, j]
            dp_pipeline_parallel_match[j][i] = [
                (element_1, element_0) for element_0, element_1 in dp_pipeline_parallel_match[i][j]]

    pipeline_parallel_cost = []
    pipeline_parallel_path = []
    for start_node in range(way):
        tsp = open_loop_tsp(cross_partition_cost, start_node)
        cost, path = tsp.get_least_cost_route()
        pipeline_parallel_cost.append(cost)
        pipeline_parallel_path.append(path)
    dp_pipeline_parallel_cost = min(pipeline_parallel_cost)
    dp_pipeline_parallel_path = pipeline_parallel_path[pipeline_parallel_cost.index(
        dp_pipeline_parallel_cost)]

    # pipeline_parallel_cost = float('inf')
    # pipeline_parallel_path = None
    # for path in itertools.permutations(range(way)):
    #    cur_cost = 0
    #    for i in range(way - 1):
    #        cur_cost += cross_partition_cost[path[i], path[i+1]]
    #    if cur_cost < pipeline_parallel_cost:
    #        pipeline_parallel_cost = cur_cost
    #        pipeline_parallel_path = path
    # assert(dp_pipeline_parallel_cost == pipeline_parallel_cost)

    return dp_pipeline_parallel_cost, dp_pipeline_parallel_path, dp_pipeline_parallel_match

def GCMA(nodes=None, population_size=None, trails=None, mode=None, peer_delay=None,Factor = 2):
    # https://dl.acm.org/doi/10.5555/2933718.2933740
    num_devices = len(nodes)
    def five_point_crossover(parent1=None, parent2=None):
        parent1_str = [0] * num_devices
        parent2_str = [0] * num_devices
        for i in range(num_devices):
            parent1_str[parent1[i]] = i // partition_size
            parent2_str[parent2[i]] = i // partition_size

        points = list(range(num_devices))
        random.shuffle(points)
        points = points[:5]

        for point in points:
            parent2_str[point] = parent1_str[point]

        partition_sizes = [0] * way
        for partition_idx in parent2_str:
            partition_sizes[partition_idx] += 1
        for i in range(num_devices):
            if partition_sizes[parent2_str[i]] > partition_size:
                for j in range(way):
                    if partition_sizes[j] < partition_size:
                        partition_sizes[j] += 1
                        break
                partition_sizes[parent2_str[i]] -= 1
                parent2_str[i] = j
        return parent2_str

    def cyclic_partitioning(offspring=None,peer_delay=None):
        def calculate_gain_default(cur_offspring=None, locked_v_idx=None,peer_delay=None):
            partition_sizes = [0] * way
            for partition_idx in cur_offspring:
                partition_sizes[partition_idx] += 1

            gain = np.zeros(shape=(num_devices, way))
            for v_idx, partition_idx in enumerate(cur_offspring):
                if locked_v_idx[v_idx] == 0:
                    gain[v_idx][partition_idx] = np.inf
                    for target_idx, target_partition_idx in enumerate(cur_offspring):
                        partial_pipeline_parallel_cost = peer_delay[v_idx, target_idx]
                        if partition_idx != target_partition_idx:
                            gain[v_idx][target_partition_idx] += partial_pipeline_parallel_cost / \
                                partition_sizes[target_partition_idx]
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

        

        def move_cycles(offspring=None,peer_delay=None):
            sum = [0]
            locked_partition_idx = [0] * way
            locked_v_idx = [0] * num_devices
            offsprings = [offspring]
            for _ in range(way):  # how many cycles
                cur_offspring = offsprings[-1].copy()
                movements = []
                epsilon = []
                tau = []
                if True:
                    G_ij, G_i, G_i_trace = calculate_gain_default(
                        cur_offspring, locked_v_idx,peer_delay=peer_delay)
                else:
                    G_ij, G_i, G_i_trace = calculate_gain_baseline(
                        cur_offspring, locked_v_idx,peer_delay=peer_delay)
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
                    if True:
                        G_ij, G_i, G_i_trace = calculate_gain_default(
                            cur_offspring, locked_v_idx,peer_delay=peer_delay)
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
            offspring = move_cycles(offspring,peer_delay=peer_delay)
        return offspring

    candidate_partitions = []
    candidate_scores = []
    candidate_min_scores = []
    for i in range(population_size):
        cur_nodes = nodes.copy()
        random.seed = i
        random.shuffle(cur_nodes)
        candidate_partitions.append(cur_nodes)

    for i, candidate_partition in enumerate(candidate_partitions):
        candidate_partition = [candidate_partition[i: i + partition_size]
                               for i in range(0, num_devices, partition_size)]
        data_parallel_cost = compute_data_parallel_cost(
            candidate_partition=candidate_partition,peer_delay=peer_delay,Factor=Factor)
        pipeline_parallel_cost, pipeline_parallel_path, pipeline_parallel_match = compute_pipeline_parallel_cost(
            candidate_partition,peer_delay=peer_delay)
        candidate_scores.append(data_parallel_cost +
                                2 * pipeline_parallel_cost)
        candidate_min_scores.append(np.min(candidate_scores))

    for i in range(trails):
        np.random.seed = i
        parent1_idx, parent2_idx = np.random.randint(population_size, size=2)
        ga_offspring_str = five_point_crossover(
            candidate_partitions[parent1_idx], candidate_partitions[parent2_idx])
        offspring_str = cyclic_partitioning(ga_offspring_str,peer_delay=peer_delay)

        offspring = [[] for _ in range(way)]
        for v_idx, partition_idx in enumerate(offspring_str):
            offspring[partition_idx].append(v_idx)
        offspring_data_parallel_cost = compute_data_parallel_cost(
            candidate_partition=offspring,peer_delay=peer_delay,Factor=Factor)
        offspring_pipeline_parallel_cost, offspring_pipeline_parallel_path, offspring_pipeline_parallel_match = compute_pipeline_parallel_cost(
            offspring,peer_delay=peer_delay)
        offspring_score = offspring_data_parallel_cost + \
            2 * offspring_pipeline_parallel_cost
        offspring = list(itertools.chain.from_iterable(offspring))

        if offspring_score > max(candidate_scores[parent1_idx], offspring_score > candidate_scores[parent2_idx]):
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


def get_pipelines(candidate_partition=None, candidate_pipeline_parallel_path=None,
                  candidate_pipeline_parallel_match=None):
    candidate_pipeline = np.zeros(shape=(way, partition_size)).astype(int)
    for stage_idx, partition_idx in enumerate(candidate_pipeline_parallel_path):
        if stage_idx:
            last_partition_idx = candidate_pipeline_parallel_path[stage_idx - 1]
            bipartite_match = candidate_pipeline_parallel_match[last_partition_idx][partition_idx]
            for match in bipartite_match:
                for i in range(partition_size):
                    if candidate_pipeline[stage_idx - 1][i] == match[0]:
                        candidate_pipeline[stage_idx][i] = match[1]
        else:
            next_partition_idx = candidate_pipeline_parallel_path[stage_idx + 1]
            bipartite_match = candidate_pipeline_parallel_match[partition_idx][next_partition_idx]
            for i, match in enumerate(bipartite_match):
                candidate_pipeline[0][i] = match[0]

    for stage_idx, partition_idx in enumerate(candidate_pipeline_parallel_path):
        for i in range(partition_size):
            candidate_pipeline[stage_idx][i] = candidate_partition[partition_idx][candidate_pipeline[stage_idx][i]]
    
    return candidate_pipeline

def run_gcma(nodes,cm,Factor):
            cm = np.array(cm)
            num_devices = len(nodes)
            candidate_partitions, all_cost_records, min_cost_records = GCMA(
                nodes=nodes, population_size=24, trails=4900, mode="default",peer_delay=cm,Factor=Factor)
            candidate_partition_idx = np.argmin(all_cost_records)
            candidate_partition = [candidate_partitions[candidate_partition_idx][i: i + partition_size]
                                   for i in range(0, num_devices, partition_size)]
            data_parallel_cost = compute_data_parallel_cost(
                candidate_partition=candidate_partition,Factor=Factor,peer_delay=cm)
            pipeline_parallel_cost, pipeline_parallel_path, pipeline_parallel_match = compute_pipeline_parallel_cost(
                candidate_partition,peer_delay=cm)
            min_total_cost = data_parallel_cost + 2 * pipeline_parallel_cost

            
            
            # print("candidate partition: " + str(candidate_partition))
            # print("pipeline parallel path: " + str(pipeline_parallel_path))
            # print("total cost: " + str(min_total_cost))
            # print("data parallel cost: " + str(data_parallel_cost))
            # print("pipeline parallel cost: " + str(2 * pipeline_parallel_cost))
            
            candidate_pipeline = get_pipelines(
                candidate_partition, pipeline_parallel_path, pipeline_parallel_match)
            # x = len(candidate_partition)
            # y = len(candidate_partition[0])
            # ret = []
            # for j in range(y):
            #     tmp =[]
            #     for i in range(x):
            #         tmp.append(candidate_partition[i][j])
            #     ret.append(tmp)
            # print(ret)
            return candidate_pipeline.tolist()
            # print(np.array(candidate_pipeline)
            ip_rank_map = [0] * num_devices
            for pipeline_idx in range(partition_size):
                for stage_idx in range(way):
                    ip_rank_map[candidate_pipeline[stage_idx,
                                                   pipeline_idx]] = pipeline_idx * way + stage_idx
            # print(ip_rank_map)
            assert(np.sum(ip_rank_map) == np.sum(range(num_devices)))