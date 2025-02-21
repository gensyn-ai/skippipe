from .graph import *
from random import sample
from copy import deepcopy
def coarsen_graph(g: Graph):
    # Coarsen graph
    nodes = list(g._matching.keys())
    paired = []
    cm = g._cost_matrix
    # maximal matching (SMALLEST edge first)
    while len(nodes) > 1:
        n = sample(nodes,1)[0]
        mnv = float("inf")
        mnidx = None
        for l in range(len(cm)):
            if l == n or l not in nodes:
                continue
            if cm[n][l] < mnv:
                mnv = cm[n][l]
                mnidx = l
        if mnidx == None:
            break
        nodes.remove(mnidx)
        nodes.remove(n)
        # print(nodes)
        paired.append((n,mnidx))
    # print(paired)
    if len(nodes) == 1:
        # unpaired one node -> add as well
        paired.append((nodes[0],nodes[0]))
    coarse = Graph(0)
    for n1,n2 in paired:
        if n1 == n2:
            tmp = Graph(n1)
            tmp.add_node(g.nodes[g._matching[n1]])
            tmp.weight = g.nodes[g._matching[n1]].weight
            count1 = 0
            count2 = 0
            if isinstance(g.nodes[g._matching[n1]],Node):
                count1 = 1
            else:
                count1 = g.nodes[g._matching[n1]].properties["count"]
            tmp.properties["count"] = count1 
            coarse.add_node(tmp)
            continue
        tmp = Graph(n1)
        tmp.add_node(g.nodes[g._matching[n1]])
        tmp.add_node(g.nodes[g._matching[n2]])
        tmp.weight = g.nodes[g._matching[n1]].weight + g.nodes[g._matching[n2]].weight
        count1 = 0
        count2 = 0
        if isinstance(g.nodes[g._matching[n1]],Node):
            count1 = 1
        else:
            count1 = g.nodes[g._matching[n1]].properties["count"]
        if isinstance(g.nodes[g._matching[n2]],Node):
            count2 = 1
        else:
            count2 = g.nodes[g._matching[n2]].properties["count"]

        tmp.properties["count"] = count1 + count2
        
        coarse.add_node(tmp)
    c = 0

    for n1,n2 in paired:
        for n3,n4 in paired:
            if n1 == n3:
                continue
            max_val = 0
            for p in [(n1,n3),(n1,n4),(n2,n3),(n2,n4)]:
                # Put the maximal degree of edges between paired groups
                max_val += cm[p[0]][p[1]]
            coarse.add_edge(Edge(coarse.nodes[n1],coarse.nodes[n3],max_val,c))
            c += 1
    coarse.fill_incident_edges()
    coarse.make_cost_matrix()
    return coarse

def recursive_bisection(g, nodes_to_consider, gain_function, calc_object):
    # create two partitions of 1 group of nodes
    initial_partitions = [[],[]]
    for idx,nd in enumerate(nodes_to_consider):
        if idx < len(nodes_to_consider)//2:
            initial_partitions[0].append(nd)
        else:
            initial_partitions[1].append(nd)

    curr_obj = calc_object(initial_partitions,g)
    
    without_improvements = 0
    curr_partition = initial_partitions
    history = [curr_partition]
    # KL-aglortihm
    while True:
        curr_partition = deepcopy(curr_partition)
        marked = []
        while True:
            max_el = None
            max_impr = 0
            for idx,p in enumerate(curr_partition):
                for el in p:
                    if el in marked:
                        continue
                    gain = gain_function(el,p,nodes_to_consider,g)
                    if gain > max_impr:
                        max_impr = gain
                        max_el = (el,idx)
            if max_el == None:
                break
            el,idx = max_el
            
            curr_partition[idx].remove(el)
            curr_partition[(idx+1)%2].append(el)
            marked.append(el)
        new_obj = calc_object(curr_partition,g)
        if new_obj >= curr_obj:
            without_improvements += 1
        else:
            without_improvements = 0
        curr_obj = new_obj
        if without_improvements == 50:
            # if 50 no-improvements -> return and undo last 50 changes
            return history[0]
        history.append(curr_partition)
        history = history[-50:]

def gains(nd: int, l1: List[int],l2: List[int], g):
    # l1 = P1==P2 and l2 P1 != P2 (or all else)
    sm1 = 0
    sm2 = 0
    for el in l1:
        sm1 += g._cost_matrix[g._reverse_matching[el]][g._reverse_matching[nd]]
    for el in l2:
        if el in l1:
            continue
        sm2 += g._cost_matrix[g._reverse_matching[el]][g._reverse_matching[nd]]
    # print(sm1,sm2,nd,l1,l2)
    return sm1 - sm2

def partition_graph(g: Graph, k, total_size: int, gain_function = None, calc_object=None):

    if not gain_function:
        # use gian function
        gain_function = gains
    partitions = [list(g.nodes.keys())]
    flg = True
    # keep bisecting until we have desired number of k-partitions
    while flg:
        flg = False
        if len(partitions) == k + 1:
            break
        for idx, p in enumerate(partitions):
            sz = 0
            for el in p:
                sz += g.nodes[el].properties["count"]
            if sz > total_size//k and len(p) > 1:
                
                ret = recursive_bisection(g,p,gain_function, calc_object)
                del partitions[idx]
                partitions += ret
                flg = True
                break
    return partitions
    
def calc_object(partitions,g):
    obj = 0
    for p in partitions:
        curr_cost = 0
        if p == partitions[-1]:
            continue
        for t in p:
            for t2 in p:
                if t == t2:
                    continue
                curr_cost += g._cost_matrix[g._reverse_matching[t2]][g._reverse_matching[t]]
        obj = max(obj,curr_cost)
    return obj

def degree(nd: int, l1: List[int], g):
    sm1 = 0
        
    for el in l1:
            
        sm1 += g._cost_matrix[g._reverse_matching[el]][g._reverse_matching[nd]]
        
    return sm1

def k_way_partition_with_r(g: Graph, k):
    
    sz = len(g.nodes)
    for v in g.nodes.values():
        v.properties["count"] = 1
    coaresened_history = [g]
    remainder = sz % k
    constraints = [(sz // k)/sz for _ in range(k)]
    constraints.append(remainder/sz)
    curr_size = sz
    
    while curr_size / 2 >= k:
        # keep coarsening
        coaresened_history.append(coarsen_graph(coaresened_history[-1]))
        curr_size = len(coaresened_history[-1].nodes)
        
    # initial partitions
    partitions = partition_graph(coaresened_history[-1],k,sz,calc_object=calc_object)
    print(partitions)
    
    
    i = len(coaresened_history) - 2
    while i >= 0:
        # uncoarsen
        print("------")
        
        g_plus_1 = coaresened_history[i + 1]
        g_0 = coaresened_history[i]
        tmp_p = []
        # project:
        for p in partitions:
            tmp = []
            for el in p:
                
                v =  g_plus_1.nodes[el]
                for nd in v.nodes.values():
                    tmp.append(nd.idx)
            tmp_p.append(tmp)
        print(tmp_p)
        partitions = tmp_p
        partition_size = []
        for p in partitions:
            tmp = 0
            for nd in p:
                tmp += g_0.nodes[nd].properties["count"]
            partition_size.append(tmp)
        smsz = sum(partition_size)
        flg = True
        # GR as described?
        while flg:
            flg = False
            unmarked = list(g_0.nodes.values())
            unmarked = [el.idx for el in unmarked]
            while len(unmarked) > 0:
                # random visit:
                # all nodes are boundaries :/
                el = sample(unmarked,1)[0]
                idv = 0
                idix = -1
                gains = []
                for ix, p in enumerate(partitions):
                    # print(el,p)
                    # compute degrees (EDb - ID)
                    gains.append(degree(el,p,g_0))
                    if el in p:
                        idv = gains[-1]
                        idix = ix
                max_gain  = 0
                
                max_idx = None
                for ix, gn in enumerate(gains):
                    # print(constraints[idix])
                    if ix == idix:
                        continue
                    
                    if idv - gn > max_gain:
                        # weight constraints:
                        if abs((g_0.nodes[el].properties["count"] + partition_size[ix])/smsz - constraints[ix]) > 0.02 * i:
                            
                            continue
                        
                        max_idx = ix
                        max_gain = idv-gn
                


                if max_idx != None:
                    
                    flg = True
                    partitions[idix].remove(el)
                    partition_size[idix] -= g_0.nodes[el].properties["count"]
                    partitions[max_idx].append(el)
                    partition_size[max_idx] += g_0.nodes[el].properties["count"]
                

                unmarked.remove(el)
            curr_obj = calc_object(partitions,g_0)
            print("CURRENT", curr_obj)
            print(partition_size)
            for ix, p in enumerate(partitions):
                # fix sizes
                if partition_size[ix]/smsz - constraints[ix] < 0.02*i:
                    print("TOO SMALL")
                    unmarked = list(g_0.nodes.values())
                    unmarked = [el.idx for el in unmarked]
                    best_gain = float("-inf")
                    best_partition_gain = 0
                    best_idx = None
                    best_p = None
                    while len(unmarked) > 0:
                        # print("consider")
                        el = sample(unmarked,1)[0]
                        unmarked.remove(el)
                        
                        idix = -1
                        
                        for ix2, p in enumerate(partitions):
                            if el in p:
                                idix = ix2
                        if idix == ix:
                            
                            continue
                        if not (partition_size[idix]/smsz - constraints[idix] >  0.02*i):
                            continue
                        # compute the gains of partition sizes
                        curr_partition_gain_1 = abs(partition_size[ix]/smsz - constraints[ix])
                        curr_partition_gain_2 = abs(partition_size[idix]/smsz - constraints[idix])
                        new_partition_gain_1 = abs((partition_size[ix] + g_0.nodes[el].properties["count"])/smsz - constraints[ix])
                        new_partition_gain_2 = abs((partition_size[idix] - g_0.nodes[el].properties["count"])/smsz - constraints[idix])
                        # compute the objective:
                        partitions[idix].remove(el)
                        partitions[ix].append(el)
                        new_obj = calc_object(partitions,g_0)
                        partitions[ix].remove(el)
                        partitions[idix].append(el)
                        # print((curr_partition_gain_1 + curr_partition_gain_2) - (new_partition_gain_1 + new_partition_gain_2))
                        if curr_obj - new_obj > best_gain and (curr_partition_gain_1 + curr_partition_gain_2) - (new_partition_gain_1 + new_partition_gain_2) >= best_partition_gain:
                            best_partition_gain = (curr_partition_gain_1 + curr_partition_gain_2) - (new_partition_gain_1 + new_partition_gain_2)
                            best_gain = curr_obj - new_obj
                            best_idx = el
                            best_p = idix
                    if best_idx != None:
                        print(partition_size)
                        flag = True
                        partitions[best_p].remove(best_idx)
                        partitions[ix].append(best_idx)
                        partition_size[best_p] -= g_0.nodes[el].properties["count"]
                        partition_size[ix] += g_0.nodes[el].properties["count"]
                        print("lack of balance can be fixed!",partition_size)
        i -= 1
        print("------")
    print(partitions)
    print(calc_object(partitions,g))
    print(partition_size)
    return partitions
    


    


    

    

    



