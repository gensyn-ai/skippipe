
import numpy as np
from schedulers.graph import *
from schedulers.a_star_modified import Agent
from schedulers.CBS import CBS, CBS_item
from schedulers.graph_partitioning import *
from schedulers.gcma import *
import random
import numpy as np
from schedulers.DTFM import run_gcma
from schedulers.com_sym import *
from schedulers.communication_costs import *
random.seed(5) # recommended is 5 as that is what we used in our experiments
np.random.seed(5)

# ----- MODIFY FROM HERE ------
PAT_LENGTH = 4
memory = 3 # memory per device


LAYERS_PER_DEVICE = 3
SAMPLES_IN_MB = 1
MB_COUNT = 6
NUMBER_OF_NODES = 20


DP_SIZE_IN_BYTES = 1346446748

# 1 sample activation size:
MB_SIZE_IN_BYTES = 16777324

# 33,554,538
FACTOR = DP_SIZE_IN_BYTES/(MB_SIZE_IN_BYTES*SAMPLES_IN_MB*MB_COUNT)
partition_sizes = [5,3,3,3,3,3]
MAX_MB_PER_STAGE = partition_sizes[1] * memory

assert sum(partition_sizes) == NUMBER_OF_NODES


setting = "geo-distributed"
# options for setting:
# geo-distributed
# single-cluster
# 5-clusters
# ----- TO HERE -------
locations = get_locations(setting)
computations = get_computations(setting)
# get nodes:
node_list = []
while len(node_list) < NUMBER_OF_NODES:
    for v in locations:
        print(len(node_list),v)
        node_list.append(v)
        if len(node_list) == NUMBER_OF_NODES:
            break


# create cost matrix and computation array
cost_matrix = [[0 for _ in range(len(node_list))] for _ in range(len(node_list))]


wm = [0 for _ in range(len(node_list))]
for x in range(len(node_list)):
    wm[x] = computations[node_list[x]]*LAYERS_PER_DEVICE*SAMPLES_IN_MB
    
    for y in range(len(node_list)):
        if x == y:
            continue
        cost_matrix[x][y] = delay_map(node_list[x],node_list[y],sz=MB_SIZE_IN_BYTES*SAMPLES_IN_MB)+0.13+0.2+0.02 # there are some additional delays due to cpu to gpu communication...
        # these are really trick to measure and need per set up accurate measurements :/
        # and unfortunately our solution does depend on accurate profiling... so good luck :)


g = Graph(0)
g2 = Graph(2)
output = {}
g.add_cost_matrix(cost_matrix,wm)
g.fill_incident_edges()
cost_matrix2 = [[1 for _ in range(len(node_list))] for _ in range(len(node_list))]


wm2 = [1 for _ in range(len(node_list))]
g2.add_cost_matrix(cost_matrix2,wm2)
g2.fill_incident_edges()
bst = None 
score = float("inf")
# Find best arrangement:
for _ in range(1):
    partitions, scores, _ = GCMA(g,partition_sizes=partition_sizes,trails=8000,population_size=200,factor=FACTOR)
    ret = np.argmin(scores)
    if scores[ret] < score:
        score = scores[ret]
        bst = reconstruct_partition(g2,partitions[ret],partition_sizes)
        bst = reconstruct_partition(g,partitions[ret],partition_sizes)
# bst = run_gcma(list(range(len(node_list)))[:-2],cost_matrix,FACTOR)
# bst[0] += list(range(len(node_list)))[-2:]
# print(bst)
# reconstruct arrangement of nodes: 
ret = bst
output["delays"] = cost_matrix
output["memory"] = memory
output["partitions"] = ret
output["GCMAscore"] = score
output["locations"] = node_list
MAIN_WM = wm
nds = {}
for idx in range(len(node_list)):
    nds[idx] = ComNode(MAIN_WM[idx],idx,3)
output["baseline-sends"] = MB_COUNT + (partition_sizes[0]-partition_sizes[1])*MB_COUNT/partition_sizes[1]
output["ours-sends"] = MB_COUNT
for num,idx in enumerate(ret[0]):
    if num >= partition_sizes[1]:
        break
    
    for k in range(3):
        for d in range(3):
            path = {}
            for p in range(1,len(ret)):
                path[ret[p-1][num]] = ret[p][num]
            mb = MB(100*d + 3*k + 9*num,path,idx)
            # print(d+3*k+15*num,idx)
            mb.tm = d*0.1
            mb.tm_receive = 10000
            nds[idx].receive(mb)
            # print(idx,len(nds[idx].received))
output["baseline-expected-time"] = run_simulation(nds,ret,cost_matrix)
for k,v in nds.items():
    # print(k,v.processed)
    if k in output["partitions"][0]:
        
        assert v.processed_total == 9 or v.processed_total == 0
            
print("EXPECTED TIME STANDARD",output["baseline-expected-time"])

tmp = []
for idx, p in enumerate(ret):
    for nd in p:
        tmp.append(nd)
        g.nodes[nd].properties["partition"] = idx
        g2.nodes[nd].properties["partition"] = idx
tmp.sort()

import json
with open(f"2_communication_{SAMPLES_IN_MB}_samples_llama_1_5b.json","w") as fd:
    json.dump(output,fd,indent=2)

# COLLISION AWARE:
agents = []
paths_in_coarsened = 3 # for coarsening change the value


delta = 3//paths_in_coarsened
for num,idx in enumerate(ret[0]):
    
    for k in range(delta):
        
        # add microbatch/agent
        agents.append(Agent(k + delta*num, idx, k*wm[idx]))
print(len(agents))
# Run CBS
best_time_ca = float("inf")
final_solutions = []

final_solutions: List[CBS_item] = CBS(g,agents,lambda x1,x2: cost_matrix[x1][x2],ret,path_l=PAT_LENGTH,memory=memory//(3//delta),mb_per_stage_max=MAX_MB_PER_STAGE//(3//delta),delta=(3//delta))
for solutions in final_solutions:
    visits_per_node = {}
    for ag_sol in solutions.solution:

        for nd in ag_sol[1]:
                    
            if nd[0] not in visits_per_node:
                visits_per_node[nd[0]] = []
            visits_per_node[nd[0]].append((ag_sol[2],nd[1])) 
    for v in visits_per_node.values():
        v.sort(key=lambda el: el[1])
    output["ca-processing-order"] = visits_per_node

    nds = {}
    for idx in range(len(node_list)):
        nds[idx] = ComNode(MAIN_WM[idx],idx,memory)

    paths = {}
    
    for ag in solutions.solution:
        
        for mb_c in range(3//delta):
            for d in range(2):
                
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
                # print(mb_c, d, ag[2], path, d*100+mb_c*10 + ag[2])
                # for t in ag[1]:
                #     print(t)
                # print(path)
                # print(ag[2], agents[ag[2]].start_idx)
                tmp = MB(d*100+mb_c*10 + ag[2],path,agents[ag[2]].start_idx)
               
                tmp.tm = d*2.5 + mb_c*1 + agents[ag[2]].dt
                tmp.tm_receive = 10000
                paths[ag[2]] = path
                nds[agents[ag[2]].start_idx].receive(tmp)
    visits_per_node = {}
    tmp = run_simulation(nds,ret,cost_matrix)
    if best_time_ca > tmp:
        best_time_ca = tmp
        output["ca-expected-time"] = tmp
        print("EXPECTED TIME WITH COLLISION AWARENESS:", tmp)
        for k,v in nds.items():
            if k in output["partitions"][0]:
                
                assert len(v.received_sent) == 6
                #
            visits_per_node[k] = len(v.received_sent)
        
        output["ca-mb-per-node"] = visits_per_node
        output["ca-paths"] = paths
    



# non-ca-aware
agents = []
for num,idx in enumerate(ret[0]):

    for k in range(memory):
        agents.append(Agent(k + memory*num, idx, k*wm[idx]))

solutions: CBS_item = CBS(g,agents,lambda x1,x2: cost_matrix[x1][x2],ret,constraints=[True,True,False],path_l=PAT_LENGTH,memory=memory,mb_per_stage_max=MAX_MB_PER_STAGE)
visits_per_node = {}
for ag_sol in solutions.solution:

    for nd in ag_sol[1]:
                
        if nd[0] not in visits_per_node:
            visits_per_node[nd[0]] = []
        visits_per_node[nd[0]].append((ag_sol[2],nd[1])) 
for v in visits_per_node.values():
    v.sort(key=lambda el: el[1])
output["non-ca-processing-order"] = visits_per_node
nds = {}
for idx in range(len(node_list)):
    nds[idx] = ComNode(MAIN_WM[idx],idx,memory)

paths = {}
for ag in solutions.solution:
    for d in range(2):
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
        # print(path)
        # print(ag[2], agents[ag[2]].start_idx)
        tmp = MB(d*100 + ag[2],path,agents[ag[2]].start_idx)
        tmp.tm = d*2.5 + agents[ag[2]].dt
        tmp.tm_receive = 10000
        paths[ag[2]] = path
        nds[agents[ag[2]].start_idx].receive(tmp)


output["nonca-expected-time"] = run_simulation(nds,ret,cost_matrix)
print("EXPECTED TIME WITHOUT COLLISION AWARENESS:", output["nonca-expected-time"])

visits_per_node = {}

for k,v in nds.items():
    visits_per_node[k] = len(v.received_sent)
output["non-ca-mb-per-node"] = visits_per_node
output["non-ca-paths"] = paths



# random AWARE:
agents = []
paths_random = []


# print(possible_paths)

agents = []
for num,idx in enumerate(ret[0]):

    for k in range(3):
        agents.append(Agent(k + 3*num, idx, k*wm[idx]))

solutions: CBS_item = CBS(g2,agents,lambda x1,x2: cost_matrix[x1][x2],ret,constraints=[True,True,False],path_l=PAT_LENGTH,memory=memory,mb_per_stage_max=MAX_MB_PER_STAGE,limit_TC1=True)
visits_per_node = {}
for ag_sol in solutions.solution:

    for nd in ag_sol[1]:
                
        if nd[0] not in visits_per_node:
            visits_per_node[nd[0]] = []
        visits_per_node[nd[0]].append((ag_sol[2],nd[1])) 
for v in visits_per_node.values():
    v.sort(key=lambda el: el[1])
output["random-processing-order"] = visits_per_node
nds = {}
for idx in range(len(node_list)):
    nds[idx] = ComNode(MAIN_WM[idx],idx,3)

paths = {}
for ag in solutions.solution:
    for d in range(2):
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
        # print(path)
        # print(ag[2], agents[ag[2]].start_idx)
        tmp = MB(d*100 + ag[2],path,agents[ag[2]].start_idx)
        tmp.tm = d*2.5 + agents[ag[2]].dt
        tmp.tm_receive = 10000
        paths[ag[2]] = path
        nds[agents[ag[2]].start_idx].receive(tmp)


output["random-expected-time"] = run_simulation(nds,ret,cost_matrix)
print("EXPECTED TIME RANDOM", output["random-expected-time"])
for k,v in nds.items():
            # if k in output["partitions"][0]:
                # assert len(v.received_sent) == 6
                # print(nds[k].processed)
            visits_per_node[k] = len(v.received_sent)

visits_per_node = {}

# for k,v in nds.items():
#     print(k,v.processed)
output["random-mb-per-node"] = visits_per_node
output["random-paths"] = paths

# save to JSON
import json
with open(f"2_communication_{SAMPLES_IN_MB}_samples_llama_1_5b.json","w") as fd:
    json.dump(output,fd,indent=2)