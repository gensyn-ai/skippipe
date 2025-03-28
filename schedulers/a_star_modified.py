from .graph import *
from typing import Dict, List, Union, Tuple, Any
import numpy as np
import heapq
from dataclasses import dataclass, field

@dataclass(order=True)
class a_star_item_modified:
    heuristic: float
    time: float
    edges_so_far: float=field(compare=False)
    reachedfrom: int=field(compare=False)
    idx: int=field(compare=False)
    edg: Edge=field(compare=False)
    back: bool=field(compare=False)
    path: List=field(compare=False)

class Agent():
    def __init__(self,idx,start_idx,dt = 0):
        self.idx = idx
        self.start_idx = start_idx
        self.dt = dt


class Conflict():
    def __init__(self, agidx: int, ndix: int,tmstart: int, tmend: int, tp: int):
        self.type = tp
        self.agidx = agidx
        self.ndix = ndix
        self.tmstart = tmstart
        self.tmend = tmend
        
        
    
    
   

def a_star_modified(g: Graph, start_idx, heuristic, agidx, dt = 0, conflicts: List[Conflict] = [], visitable = dict(), visitable_stages = dict(), lng = 2):
    h: List[a_star_item_modified] = []
    
    lng -= 1
    max_partition = 0
    for nd in g.nodes.values():
        max_partition = max(nd.properties["partition"],max_partition)
    # print(max_partition)
    heapq.heapify(h)
    heapq.heappush(h,a_star_item_modified(dt,dt,0,start_idx,start_idx,None,False,[]))
    
    def conflict_check(conflict: Conflict, time, weight):
        # check for conflicts
        if conflict.type != 3:
            # if not type 3 conflict - no need to check
            return time
        # print("TYPE 3?",conflict.agidx,conflict.ndix,time ,weight,conflict.tmstart,conflict.tmend)
        if (time >= conflict.tmstart and time < conflict.tmend) or (time + weight >= conflict.tmstart and time + weight < conflict.tmend):
            # our start time is between the conflict start time and end time or our end time is between them - conflict
            # print("TYPE 3!!!",conflict.agidx,conflict.ndix,time ,weight,conflict.tmstart,conflict.tmend)
            return conflict.tmend
        elif (time <= conflict.tmstart and time + weight > conflict.tmstart) or (time <= conflict.tmend and time + weight > conflict.tmend):
            # the conflict's start time is between our time start and time end or the conflicts time end is between them - conflict
            return conflict.tmend # on conflict we delay our time to the conflcit time end
        else:
            return time # no conflict current time

    def reconstruct_path(path_edges: List[Edge],frm,dt):
        # RECONDSTURCT THE PATH ONCE WE HAVE IT. COST OF THE PATH IS FORWARD + BACKWARD (COMPUTATIONS AND COMMUNICATIONS TAKEN INTO ACCOUNT)
        path: List[Tuple[idx,Edge]] = []
        el: Tuple[idx,Edge]= (frm,None)
        node_visits = []
        path.append(el)
        curr = frm
        for edg in path_edges:
            if edg.n1.idx == curr:
                path.append((edg.n2.idx , edg))
                curr = edg.n2.idx
            else:
                path.append((edg.n1.idx , edg))
                curr = edg.n1.idx
            
        
        t = dt
        prv = None

        for el,edg in path:
            max_offset = t
            if edg:
                max_offset += edg.w
            flg = True
            process_time = g.nodes[el].weight
            if el == frm and edg != None:
                process_time = (process_time)/4
            while flg:
                flg = False
                for c in conflicts:
                    
                    if c.agidx == agidx and c.ndix == el:
                        tmp_m = max_offset
                        max_offset = max(conflict_check(c,max_offset,process_time),max_offset)
                        if max_offset > tmp_m:
                            flg = True
                    

                        
            
            t = max_offset
            
            t += process_time
            node_visits.append((el,max_offset,t))
        

        path.reverse()
        flg = True
        for el,edg in path:
            if flg:
                t += edg.w
                flg = False
                continue
            max_offset = t
            if not prv:
                
                prv = el
            else:
                prv = el
            flg = True
            while flg:
                flg = False
                for c in conflicts:
                    
                    if c.agidx == agidx and c.ndix == el:
                        tmp_m = max_offset
                        max_offset = max(conflict_check(c,max_offset,1.5*(g.nodes[el].weight)),max_offset)
                        if max_offset > tmp_m:
                            flg = True
            
            t = max_offset
            
            t += 1.5*(g.nodes[el].weight)
            node_visits.append((el,max_offset,t))
            if edg:
                t+=edg.w

        return node_visits,t
    prv_dist = 0
    while len(h) > 0:
        
        el = heapq.heappop(h)
       
        t = el.time
       
        
        prv_dist = t 
        if visitable[agidx][el.idx] == 0:
            if el.idx == start_idx:
                print("ISSUE!??!?!")
                exit()
        
            continue
        
        
        if el.idx == start_idx and el.back:
            # if we made it back - this is the solution
            return el.time, el.path, agidx
        
        
        max_offset = t # check for conflicts
        for c in conflicts:
            # check if concflict with current time at given location
            if c.agidx == agidx and c.ndix == el.idx:
                if c.ndix != start_idx or len(el.path) == 0:

                    max_offset = max(conflict_check(c,max_offset,g.nodes[el.idx].weight),max_offset)
                else:
                    max_offset = max(conflict_check(c,max_offset,(g.nodes[el.idx].weight)/4),max_offset)

                    

        if max_offset > t:
            # conflict has occured
            if max_offset == float("inf"):
                continue
            # requeue this node with the proper delay
            # max_offset - t gives us the time delay we need to wait outside of this node

            heapq.heappush(h,a_star_item_modified((max_offset - t) + el.heuristic,max_offset, el.edges_so_far, el.reachedfrom,el.idx,el.edg,el.back,el.path))
            continue

        
        

        if el.idx == start_idx and len(el.path) != 0:
            # We reached end node, now let's do the backwards pass:
            path,t = reconstruct_path(el.path,start_idx,dt)

            heapq.heappush(h,a_star_item_modified(t,t,el.edges_so_far,None,start_idx,None,True,path))
            continue

        # time gets increased with amount of time needed to process mb in forward
        el.time += g.nodes[el.idx].weight
        number_of_swaps = 0 # count how many times we swapped
        frontier = 0 # furthest stage visited so far
        partitions = [0] # partitions visited
        furthest_swap  = 0
        number_of_skips = 0
        count_swaps = 0
        curr = start_idx
        for edg in el.path:
            prev_frontier = frontier
            if edg.n1.idx == curr:
                
                frontier = max(frontier,edg.n2.properties["partition"])
                
                curr = edg.n2.idx
            else:
                frontier = max(frontier,edg.n1.properties["partition"])
                
                curr = edg.n1.idx
            number_of_skips += (frontier - prev_frontier) - 1
            if g.nodes[curr].properties["partition"] < frontier: # a swap has occurred 
                furthest_swap = max((frontier-g.nodes[curr].properties["partition"]),furthest_swap)
                count_swaps += 1
                number_of_swaps += (frontier-g.nodes[curr].properties["partition"]) # the distance of the swap
            partitions.append(g.nodes[curr].properties["partition"])
        if number_of_swaps > 2 or furthest_swap > 1 or count_swaps > 1:
            
            continue
        # if number_of_skips > max_partition+1 - lng:
        #     continue

        for edg in g.nodes[el.idx].incident_edges.values():
            # print(len(g.nodes[el.idx].incident_edges.values()))
            if edg.directed: # dealing with directed graphs, we dont care
                if edg.n1.idx != el.idx:
                    continue
                if len(el.path) < lng and edg.n2.idx == start_idx:
                    continue
                elif len(el.path) == lng and edg.n2.idx == start_idx:
                    heapq.heappush(h,a_star_item_modified(el.time + 2*edg.w + 1.5*g.nodes[el.idx].weight + el.edges_so_far, el.time + edg.w, el.edges_so_far + edg.w + 1.5*g.nodes[el.idx].weight, el.idx,edg.n2.idx,edg,False,el.path.copy() + [edg]))
                    continue
                if len(el.path) == lng:
                        continue
                if visitable_stages[agidx][g.nodes[edg.n2.idx].properties["partition"]] == 0:
                    continue
                if number_of_swaps > 3 and g.nodes[edg.n2.idx].properties["partition"] < frontier:
                    continue
                if g.nodes[edg.n2.idx].properties["partition"] in partitions:
                    
                    continue
                if abs(g.nodes[edg.n2.idx].properties["partition"] - frontier) > 2:
                    continue
                heapq.heappush(h,a_star_item_modified(el.time + 2*edg.w + 1.5*g.nodes[el.idx].weight + el.edges_so_far, el.time  + edg.w,el.edges_so_far + edg.w + 1.5*g.nodes[el.idx].weight,el.idx,edg.n2.idx,edg,False,el.path.copy() + [edg]))
            else: # undirected graphs (our case)
                if edg.n1.idx == el.idx:
                    if len(el.path) < lng and edg.n2.idx == start_idx: # if we dont have desired path length and it is start - ignore
                        continue

                    elif len(el.path) == lng and edg.n2.idx == start_idx:  # if desired path length and end node - MUST TAKE IT
                        heapq.heappush(h,a_star_item_modified(el.time + 2*edg.w + 1.5*g.nodes[el.idx].weight + el.edges_so_far, el.time + edg.w, el.edges_so_far + edg.w + 1.5*g.nodes[el.idx].weight, el.idx,start_idx,edg,False,el.path.copy() + [edg]))
                        continue
                    if len(el.path) == lng: # path length has been reached but we are not considering start... continue
                        continue
                    if number_of_swaps > 1 and g.nodes[edg.n2.idx].properties["partition"] < frontier: # we will exceed our budget!
                        continue
                    if visitable_stages[agidx][g.nodes[edg.n2.idx].properties["partition"]] == 0 :
                        
                        continue
                    if g.nodes[edg.n2.idx].properties["partition"] in partitions:
                        # print("cannot visit",edg.n2.idx,g.nodes[edg.n2.idx].properties["partition"],partitions)
                        continue
                    if abs(g.nodes[edg.n2.idx].properties["partition"] - frontier) > 4: # skipping more than 4
                        
                        continue
                    heapq.heappush(h,a_star_item_modified( el.time + 2*edg.w + 1.5*g.nodes[el.idx].weight + el.edges_so_far, el.time + edg.w,el.edges_so_far + edg.w + 1.5*g.nodes[el.idx].weight,el.idx,edg.n2.idx,edg,False,el.path.copy() + [edg]))
                else: # same as before but for other edge end
                    if len(el.path) < lng and edg.n1.idx == start_idx:
                        
                        continue
                    elif len(el.path) == lng and edg.n1.idx == start_idx:
                        heapq.heappush(h,a_star_item_modified(  el.time + 2*edg.w  + 1.5*g.nodes[el.idx].weight + el.edges_so_far, el.time + edg.w,el.edges_so_far + edg.w + 1.5*g.nodes[el.idx].weight,el.idx,start_idx,edg,False,el.path.copy() + [edg]))
                        continue
                    if len(el.path) == lng:
                        continue
                    if number_of_swaps > 1 and g.nodes[edg.n1.idx].properties["partition"] < frontier:
                        continue
                    if visitable_stages[agidx][g.nodes[edg.n1.idx].properties["partition"]] == 0:
                        continue
                    if g.nodes[edg.n1.idx].properties["partition"] in partitions:
                        
                        continue
                    if abs(g.nodes[edg.n1.idx].properties["partition"] - frontier) > 4:
                        
                        continue
                    
                    heapq.heappush(h,a_star_item_modified( el.time + 2*edg.w + 1.5*g.nodes[el.idx].weight + el.edges_so_far,el.time + edg.w,el.edges_so_far + edg.w + 1.5*g.nodes[el.idx].weight,el.idx,edg.n1.idx,edg,False,el.path.copy() + [edg]))
    # print(agidx,"didnt find solution")
    return None

