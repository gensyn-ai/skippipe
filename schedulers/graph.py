from typing import Dict, List, Union, Tuple, Any

class Node():
    def __init__(self,idx,weight = 0, **kwargs):
        self.weight = weight
        self.idx = idx
        self.incident_edges = dict()
        self.properties = kwargs
        

class Edge():
    def __init__(self, n1:Node, n2: Node, w: float, idx: int, directed = False, **kwargs):
        self.n1 = n1
        self.n2 = n2
        self.w = w
        self.directed = directed
        self.idx = idx
        self.properties = kwargs


class HyperEdge():
    def __init__(self,edgs: List[Edge], idx: int, **kwargs):
        self.edges = edge
        self.idx = idx
        self.properties = kwargs


class Graph():
    def __init__(self, idx, **kwargs):
        self.idx = idx
        self.nodes: Dict[int, Union[Node,Graph]] = dict()
        self.edges: Dict[int, Union[Edge,HyperEdge]] = dict()
        self.incident_edges = dict()
        self.properties = kwargs
        if "weight" in kwargs:
            self.weight = kwargs["weight"]
        else:
            self.weight = 0
    
    def fill_incident_edges(self):
        for k,v in self.edges.items():
            v.n1.incident_edges[k] = v
            if not v.directed:
                v.n2.incident_edges[k] = v
        
    def add_cost_matrix(self, cm, wm = None, keep_self_edges = False):

        self._cost_matrix = cm
        self._wm = wm
        self._matching = dict()
        self.nodes: Dict[int, Union[Node,Graph]] = dict()
        self.edges: Dict[int, Union[Edge,HyperEdge]] = dict()
        self._reverse_matching = self._matching
        count = 0
        for x in range(len(cm)):
            if x not in self.nodes:
                self.nodes[x] = Node(x)
                self._matching[x] = x
                if wm:
                    self.nodes[x].weight = wm[x]
                    
                
            for y in range(len(cm)):
                if not keep_self_edges and x==y:
                    continue

                if y not in self.nodes:
                    self.nodes[y] = Node(y)
                    self._matching[y] = y
                    if wm:
                        self.nodes[y].weight = wm[y]
                        
                if cm[x][y] == None:
                    continue
                if cm[x][y] == cm[y][x]:
                    if x > y:
                        continue
                    self.edges[count] = Edge(self.nodes[x],self.nodes[y],cm[x][y],count) 
                    
                else:
                    self.edges[count] = Edge(self.nodes[x],self.nodes[y],cm[x][y],count,directed=True) 
                count += 1

    def add_node(self,n: Node, override_index = None):
        if override_index == None:
            override_index = n.idx
        self.nodes[override_index] = n
        return self
    
    def add_edge(self, e: Union[Edge,HyperEdge]):
        self.edges[e.idx] = e
        return self
    
    def make_cost_matrix(self):
        self._matching: Dict[int, int] = dict()
        self._reverse_matching: Dict[int, int] = dict()
        idx = 0
        for k,v in self.nodes.items():
            self._matching[idx] = k
            self._reverse_matching[k] = idx
            # print(idx,"to",k)
            idx+=1
        self._cost_matrix = [[None if x != y else 0 for x in range(idx)] for y in range(idx)]
        self._wm = [None for x in range(idx)]
        for k,v in self.edges.items():
            
            id1 = v.n1.idx
            id2 = v.n2.idx
            w = v.w
            id1 = self._reverse_matching[id1]
            id2 = self._reverse_matching[id2]
            self._wm[id1] = v.n1.weight
            self._wm[id2] = v.n2.weight
            self._cost_matrix[id1][id2] = w
            if v.directed:
                self._cost_matrix[id2][id1] = w

        return self._cost_matrix