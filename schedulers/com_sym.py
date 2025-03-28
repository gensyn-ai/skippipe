from collections import OrderedDict
from math import floor
from copy import deepcopy
class MB(object):
    def __init__(self, uniqid, path, first_node):
        self.tm = 0
        self.uniqid = uniqid
        self.path = path
        self.processed = False
        self.tm_receive = None
        self.back = False
        self.first_node = first_node
        self.one_time = False
        self.invpath = {}
        for k,v in path.items():
            self.invpath[v] = k
        # print(self.path)
        # print(self.invpath)

class ComNode(object):
    def __init__(self,comp, ndid, memory):
        
        self.comp = comp
        self.ndid = ndid
        self.received = []
        self.backreceived = []
        self.processed = []
        self.collisions = []
        self.process_at = []
        self.processed_total = 0
        self.last_tm = 0
        self.freed = []
        self.initial_sends = []
        self.memory = memory
        self.received_sent = OrderedDict()
    


    def receive(self,b: MB):
        if b.tm_receive == None:
            # b.tm += 0.02
            b.tm_receive = b.tm
        self.received.append(b)
    def backreceive(self,b:MB):
        if b.tm_receive == None:
            # b.tm += 0.02
            b.tm_receive = b.tm
        self.backreceived.append(b)
    def send(self, dl, nds, tmunit):
        self.received.sort(key=lambda mb: (mb.tm,mb.tm_receive))
        
        for idx,mb in enumerate(self.received):
            if floor(mb.tm) != tmunit:
                continue
            if mb.tm < self.last_tm:
                mb.tm = self.last_tm
                continue
            # print(self.ndid,mb.tm,tmunit)
            if mb.processed:
                continue
            
            if not mb.back and self.memory <= 0 and not mb.one_time:
                # print(self.ndid,"CANNOT TAKE IN",self.memory)
                
                mb.tm += 0.1
                continue
            if mb.first_node == self.ndid and not mb.one_time and not mb.back:
                if mb.uniqid%100 in self.freed:
                    continue
            # print(self.ndid,"processing ",mb.uniqid," at ",mb.tm, mb.back)
            
            nxt = None
            if self.ndid in mb.path:
                
                nxt = mb.path[self.ndid]
            
            v = None
            if nxt != None and not mb.back and not mb.one_time:
                self.initial_sends.append((mb.uniqid,mb.tm,mb.tm + self.comp))
                # print(self.ndid, "to",nxt)
                self.freed.append(mb.uniqid)
                self.processed.append((mb.tm, mb.tm + self.comp, mb.back, mb.uniqid))

                v = (mb.tm, mb.tm + self.comp)
                tmp = deepcopy(mb)
                tmp.tm_receive = None
                self.memory -= 1
                tmp.tm = tmp.tm + self.comp + dl[self.ndid][nxt]
                nds[nxt].receive(tmp)
            
            elif not mb.back and self.ndid in mb.invpath and not mb.one_time:
                
                self.processed.append((mb.tm, mb.tm + self.comp, mb.back, mb.uniqid))

                v = (mb.tm, mb.tm + self.comp)
                nxt = mb.first_node
                # print("last node send to", nxt)
                tmp = deepcopy(mb)
                tmp.tm_receive = None
                tmp.first_node = self.ndid
                tmp.one_time = True
                self.memory -= 1
                tmp.tm = tmp.tm + self.comp + dl[self.ndid][nxt]
                nds[nxt].receive(tmp)
            elif mb.one_time:
                # print("one time send it back")
                self.processed.append((mb.tm, mb.tm + (self.comp)/4, mb.back, mb.uniqid))

                v = (mb.tm, mb.tm + (self.comp)/4)
                nxt = mb.first_node
                tmp = deepcopy(mb)
                tmp.tm_receive = None
                tmp.first_node = self.ndid
                tmp.back = True
                tmp.one_time = False
                tmp.tm = tmp.tm + (self.comp)/4 + dl[self.ndid][nxt]
                nds[nxt].receive(tmp)

            elif self.ndid in mb.invpath:
                # print("BACK")
                self.processed.append((mb.tm, mb.tm + (self.comp)*1.5, mb.back, mb.uniqid))

                v = (mb.tm, mb.tm + (self.comp)*1.5)
                nxt = mb.invpath[self.ndid]
                self.memory += 1
                # print(self.ndid, "to",nxt)
                tmp = deepcopy(mb)
                tmp.tm_receive = None
                tmp.back = True
                tmp.tm = tmp.tm + (self.comp)*1.5 + dl[self.ndid][nxt]
                nds[nxt].receive(tmp)
            else:
                # print(self.ndid,"RECEIVED BACK",mb.tm,mb.uniqid)
                i = 0
                while i < len(self.freed):
                    if self.freed[i] == mb.uniqid%100:
                        del self.freed[i]
                        break
                    i+=1
                
                self.processed.append((mb.tm, mb.tm + (self.comp)*1.5, mb.back, mb.uniqid))
                self.processed_total += 1
                v = (mb.tm, mb.tm + (self.comp)*1.5)
                self.received_sent[mb.uniqid] = (mb.tm, mb.tm + (self.comp)*1.5, "BACK")
                self.memory += 1
            if mb.tm_receive > 1000:
                self.last_tm = v[1]+0.09
            else:
                self.last_tm = v[1]+0.13
            for idx2,mb2 in enumerate(self.received):
                if idx2 == idx:
                    continue
                if mb2.uniqid == mb.uniqid:
                    continue
                
                
                if v[0] <= mb2.tm and self.last_tm > mb2.tm:
                    # print("collision", mb2.uniqid,"postponed to",v[1],"from",mb2.tm,"period start",v[0], "on",self.ndid)
                    self.collisions.append((mb2.uniqid, v[1]-mb2.tm))
                    mb2.tm = self.last_tm
            mb.processed = True

def run_simulation(nds, partitions, cost_matrix):
    for tm in range(10000):
        count = 0
        for nd in partitions[0]: 
            count += nds[nd].processed_total
        # if count == 42:
        #     break
        for _ in range(15):

            for idx in range(len(nds)):
               nds[idx].send(cost_matrix,nds,tm)
    largest = 0
    # for nd, v in nds.items():
    #     print(nd,"received",v.processed_total)
    # for nd in range(len(nds)):
    #     for idx1, mb in enumerate(nds[nd].processed):
    #         for idx2, mb2 in enumerate(nds[nd].processed):
    #             if idx1 == idx2:
    #                 continue
    #             if mb[0] <= mb2[0] and mb[1] - 0.2 > mb2[0]:
    #                 print("ERROR")
    #                 exit()
    slowest_mb = None
    for nd in partitions[0]:    
        print(nd,"received", len(nds[nd].received_sent))
        print(nd, len(nds[nd].received_sent),nds[nd].received_sent)
        # print(nds[nd].initial_sends)
        for k,v in nds[nd].received_sent.items():
            
            largest = max(v[1],largest)
            if largest == v[1]:
                slowest_mb = k
    coll = 0
    dl = 0
    for nd in nds.values():
        
        for mb in nd.collisions:
            if mb[0] != slowest_mb:
                continue
            dl += mb[1]
            coll += 1
    # print(coll, dl)
    return largest