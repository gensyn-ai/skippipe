from .graph import *
from typing import Dict, List, Union, Tuple, Any
import numpy as np
import heapq
from dataclasses import dataclass, field

@dataclass(order=True)
class a_star_item_modified:
    
    time: int
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
    
    heapq.heapify(h)
    heapq.heappush(h,a_star_item_modified(dt,start_idx,start_idx,None,False,[]))
    
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
        # if agidx == 6:
        #     print("considering ",el.idx,start_idx)
        t = el.time
        if t < prv_dist:
            print("ERROR?")
           
            exit()
        # assert t >= prv_dist # make sure the distance (time) can only increase
        
       
        prv_dist = t 
        if visitable[agidx][el.idx] == 0:
            if el.idx == start_idx:
                print("ISSUE!??!?!")
                exit()
            # if agidx == 6:
            #     print("cannot visit",el.idx)
            #     print(visitable)
            # if not visitable (cant visit this edge), continue
            continue
        # print(t)
        
        if el.idx == start_idx and el.back:
            # if we made it back - this is the solution
            return el.time, el.path, agidx
        
        
        max_offset = t # check for conflicts
        for c in conflicts:
            # check if concflict with current time at given locatiom
            if c.agidx == agidx and c.ndix == el.idx:
                if c.ndix != start_idx or len(el.path) == 0:

                    max_offset = max(conflict_check(c,max_offset,g.nodes[el.idx].weight),max_offset)
                else:
                    max_offset = max(conflict_check(c,max_offset,(g.nodes[el.idx].weight)/4),max_offset)

                    

        if max_offset > t:
            # conflict has occured
            if max_offset == float("inf"):
                print("forbidden node!!")
                continue
            # requeue this node with the proper delay
            # max_offset - t gives us the time delay we need to wait outside of this node
            # if agidx == 6:
            #     print("delaying visit to",el.idx,t,max_offset)
            heapq.heappush(h,a_star_item_modified(max_offset, el.reachedfrom,el.idx,el.edg,el.back,el.path))
            continue
        # time gets increased wiht amount of time needed to process mb in forward
        el.time += g.nodes[el.idx].weight
        
        
        if el.idx == start_idx and len(el.path) != 0:
            
            
            # we got back - reconstruct path and record the time
            # if agidx == 6:
            #     print("END REACHED",t,agidx)
            path,t = reconstruct_path(el.path,start_idx,dt)
            # return t, path, agidx
            
            heapq.heappush(h,a_star_item_modified(t,None,start_idx,None,True,path))
            continue
        
        number_of_swaps = 0 # count how many times we swapped
        frontier = 0 # furthest stage visited so far
        partitions = [0] # partitions visited
        furthest_swap  = 0
        count_swaps = 0
        curr = start_idx
        for edg in el.path:
            
            if edg.n1.idx == curr:
                
                frontier = max(frontier,edg.n2.properties["partition"])
                
                curr = edg.n2.idx
            else:
                frontier = max(frontier,edg.n1.properties["partition"])
                
                curr = edg.n1.idx
            
            if g.nodes[curr].properties["partition"] < frontier: # a swap has occurred 
                furthest_swap = max((frontier-g.nodes[curr].properties["partition"]),furthest_swap)
                count_swaps += 1
                number_of_swaps += (frontier-g.nodes[curr].properties["partition"]) # the distance of the swap
            partitions.append(g.nodes[curr].properties["partition"])
        if number_of_swaps > 2 or furthest_swap > 1 or count_swaps > 1:
            
            continue


        for edg in g.nodes[el.idx].incident_edges.values():
            # print(len(g.nodes[el.idx].incident_edges.values()))
            if edg.directed: # dealing with directed graphs, we dont care
                if edg.n1.idx != el.idx:
                    continue
                if len(el.path) < lng and edg.n2.idx == start_idx:
                    continue
                elif len(el.path) == lng and edg.n2.idx == start_idx:
                    heapq.heappush(h,a_star_item_modified(el.time + edg.w, el.idx,edg.n2.idx,edg,False,el.path.copy() + [edg]))
                    continue
                if len(el.path) == lng:
                        continue
                if visitable_stages[agidx][g.nodes[edg.n2.idx].properties["partition"]] == 0:
                    continue
                if number_of_swaps > 3 and g.nodes[edg.n2.idx].properties["partition"] < frontier:
                    continue
                if g.nodes[edg.n2.idx].properties["partition"] in partitions:
                    
                    continue
                if abs(g.nodes[edg.n2.idx].properties["partition"] - frontier) > 3:
                    continue
                heapq.heappush(h,a_star_item_modified(el.time  + edg.w,el.idx,edg.n2.idx,edg,False,el.path.copy() + [edg]))
            else: # undirected graphs (our case)
                if edg.n1.idx == el.idx:
                    if len(el.path) < lng and edg.n2.idx == start_idx: # if we dont have desired path length and it is start - ignore
                        # if agidx == 6:
                        #     print("too small")
                        continue
                    elif len(el.path) == lng and edg.n2.idx == start_idx:  # if desired path length and end node - MUST TAKE IT
                        heapq.heappush(h,a_star_item_modified(el.time + edg.w, el.idx,start_idx,edg,False,el.path.copy() + [edg]))
                        # if agidx == 6:
                        #     print("WE NEED TO VISIT END",start_idx)
                        continue
                    if len(el.path) == lng: # path length has been reached but we are not considering start... continue
                        continue
                    if number_of_swaps > 4 and g.nodes[edg.n2.idx].properties["partition"] < frontier: # we will exceed our budget!
                        continue
                    if visitable_stages[agidx][g.nodes[edg.n2.idx].properties["partition"]] == 0 :
                        # print("cannot visit",edg.n2.idx)
                        continue
                    if g.nodes[edg.n2.idx].properties["partition"] in partitions:
                        # print("cannot visit",edg.n2.idx,g.nodes[edg.n2.idx].properties["partition"],partitions)
                        continue
                    if abs(g.nodes[edg.n2.idx].properties["partition"] - frontier) > 4: # skipping more than 4
                        # print("gap too big",frontier,g.nodes[edg.n2.idx].properties["partition"],partitions)
                        continue
                    heapq.heappush(h,a_star_item_modified( el.time + edg.w,el.idx,edg.n2.idx,edg,False,el.path.copy() + [edg]))
                else: # same as before but for other edge end
                    if len(el.path) < lng and edg.n1.idx == start_idx:
                        # if agidx == 6:
                        #     print("path length too small")
                        continue
                    elif len(el.path) == lng and edg.n1.idx == start_idx:
                        heapq.heappush(h,a_star_item_modified( el.time + edg.w,el.idx,start_idx,edg,False,el.path.copy() + [edg]))
                        # if agidx == 6:
                        #     print("WE NEED TO VISIT END",start_idx)
                        continue
                    if len(el.path) == lng:
                        continue
                    if number_of_swaps > 4 and g.nodes[edg.n1.idx].properties["partition"] < frontier:
                        continue
                    if visitable_stages[agidx][g.nodes[edg.n1.idx].properties["partition"]] == 0:
                        # print("cannot visit",edg.n1.idx)
                        continue
                    if g.nodes[edg.n1.idx].properties["partition"] in partitions:
                        # print("cannot visit",edg.n1.idx,g.nodes[edg.n1.idx].properties["partition"],partitions)
                        continue
                    if abs(g.nodes[edg.n1.idx].properties["partition"] - frontier) > 4:
                        # print("gap too big",frontier,g.nodes[edg.n1.idx].properties["partition"],partitions)
                        continue
                    
                    heapq.heappush(h,a_star_item_modified( el.time + edg.w,el.idx,edg.n1.idx,edg,False,el.path.copy() + [edg]))
    print(agidx,"didnt find solution")
    return None

