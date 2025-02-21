import numpy as np
from .graph import *
from typing import Dict, List, Union, Tuple, Any
def tsp_dp(g: Graph, start_idx: int, isOpen = False):
    cost_matrix = np.array(g._cost_matrix)
    num_nodes = cost_matrix.shape[0]
    start_idx = g._matching[start_idx]
    dp_table  = np.full(
                shape=(num_nodes, pow(2, num_nodes)), fill_value=np.inf)
    trace_table = np.zeros(
                shape=(num_nodes, pow(2, num_nodes)))
    def convert(future_nodes):
        binary_future_nodes = 0
        for future_node in future_nodes:
            binary_future_nodes += pow(2, future_node)
        return binary_future_nodes
    
    def solve(node,future_nodes,isOpen):
        if len(future_nodes) == 0:
            if isOpen:
                return 0
            else:
                return cost_matrix[node][start_idx]
        all_distance = []
        for next_node in future_nodes:
                next_future_nodes = future_nodes.copy()
                next_future_nodes.remove(next_node)
                if cost_matrix[node][next_node] == None:
                    continue
                binary_next_future_nodes = convert(next_future_nodes)
                
                if dp_table[next_node][binary_next_future_nodes] == np.inf:
                    all_distance.append(
                        cost_matrix[node][next_node] + solve(next_node, next_future_nodes,isOpen))
                else:
                    all_distance.append(
                        cost_matrix[node][next_node] + dp_table[next_node][binary_next_future_nodes])

        min_distance = min(all_distance)
        next_node = future_nodes[all_distance.index(min_distance)]

        binary_future_nodes = convert(future_nodes)
        dp_table[node][binary_future_nodes] = min_distance
        trace_table[node][binary_future_nodes] = next_node
        return min_distance
    
    future_nodes = list(range(num_nodes))
    future_nodes.remove(start_idx)
    cost = solve(start_idx, future_nodes,isOpen)

    path = [start_idx]
    cur_node = start_idx
    while len(future_nodes) > 0:
        binary_future_nodes = convert(future_nodes)
        cur_node = int(trace_table[cur_node][binary_future_nodes])
        future_nodes.remove(cur_node)
        path.append(cur_node)
    return cost, path