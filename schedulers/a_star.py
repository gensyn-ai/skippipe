from typing import Dict, List, Union, Tuple, Any
import numpy as np
import heapq
from dataclasses import dataclass, field




@dataclass(order=True)
class a_star_item:
    dist: int
    reachedfrom: int=field(compare=False)
    idx: int=field(compare=False)
    edg: Edge=field(compare=False)
def a_star(g: Graph, start_idx, goal_idx, heuristic):
    h = []
    heapq.heapify(h)
    heapq.heappush(h,a_star_item(0,start_idx,start_idx,None))
    visited: Dict[int,int] = dict()
    def reconstruct_path(dist,frm,to):
        path = [frm]
        el = frm
        while el != to:
            el = visited[el]
            path.append(el)
        path.reverse()
        return path,dist

    while len(h) > 0:
        el = heapq.heappop(h)
        if el.idx in visited:
            continue
        

        visited[el.idx] = el.reachedfrom
        if el.idx == goal_idx:
            return reconstruct_path(el.dist,el.idx,start_idx)
        for edg in g.nodes[el.idx].incident_edges.values():
            if edg.directed:
                if edg.n1.idx != el.idx:
                    continue
                heapq.heappush(h,a_star_item(el.dist + edg.w + heuristic(el.idx,edg.n2.idx),el.idx,edg.n2.idx,edg))
            else:
                if edg.n1.idx == el.idx:
                    heapq.heappush(h,a_star_item(el.dist + edg.w + heuristic(el.idx,edg.n2.idx),el.idx,edg.n2.idx,edg))
                else:
                    heapq.heappush(h,a_star_item(el.dist + edg.w + heuristic(el.idx,edg.n1.idx),el.idx,edg.n1.idx,edg))
    return None