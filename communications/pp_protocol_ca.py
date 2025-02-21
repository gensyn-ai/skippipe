from dataclasses import dataclass
import os
import random
from typing import Callable
from deccom.peers.peer import Peer
from deccom.protocols.abstractprotocol import AbstractProtocol
from deccom.protocols.wrappers import *
from datetime import datetime
import asyncio
from traceback import print_exception, format_exc
from .llm_subp import *
from time import sleep, time
from deccom.cryptofuncs.hash import SHA256
import struct
import itertools 
'''
Protocol for synchronisation between processes. Basically informs a process that it should receive from us a certain tensor :/
'''
class PPProtocl_CA(AbstractProtocol):
    required_lower = AbstractProtocol.required_lower + \
        ["find_peer", "get_peer", "get_peers", "connected_callback","disconnected_callback"]
    FORWARD_FLAG = int.from_bytes(b'\x01', byteorder="big")
    BACK_FLAG = int.from_bytes(b'\x02', byteorder="big")
    AGGREGATE_FLAG = int.from_bytes(b'\x03', byteorder="big")
    
    def __init__(self, world_size: int, stage, communication, can_process, datanodes: int, memory: int, queue_in: Queue, queue_out: Queue, subprocess:Process, submodule=None, callback: Callable[[tuple[str, int], bytes], None] = lambda : ..., 
                        MB_SEND_COUNT = 5, dp_order = 0):
        
        super().__init__(submodule, callback)
        self.group = []
        self.dp_order = dp_order
        self.p_len = None
        self.datanodes = datanodes
        self.stage = stage
        self.can_process = can_process
        self.communication = communication
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.subprocess = subprocess
        self.disconnected_callback = lambda *args : ...
        self.connected_callback = lambda *args : ...
        self.world_size = world_size
        self.memory = memory
        self.MAX_MEM = memory
        self.processed = []
        self.forwards_collected = 0
        self.mb_send = 0
        self.MB_SEND_COUNT = MB_SEND_COUNT
        self.iteration = 0
        self.received_aggregates = 0
        self.deferred = []
        self.to_receive = []
        self.send_receives = dict()
        self.saved_backs = []
        
    def run_back_scheduler(self):
        best_schedule = None
        best_time = float("inf")
        if self.forwards_collected < 8:
            loop = asyncio.get_event_loop()
            loop.create_task(self.start_back())
            return

        for comb in itertools.permutations(self.to_receive):
            tm = comb[0][1]
            worst_time = 0
            for mb in comb:
                worst_time = max(max(tm,mb[1]) + (1+mb[2])*self.compute_time, worst_time)
                tm = max(tm,mb[1]) + self.compute_time
            if worst_time < best_time:
                best_schedule = comb
                best_time = worst_time
                #it was the best of time
                #it was the worst of time
        self.to_receive = []
        for v in best_schedule:
            self.to_receive.append(v)
         
        loop = asyncio.get_event_loop()
        loop.create_task(self.start_back())  
        return

    async def start_back(self):
        
        if len(self.to_receive) == 0:
            if len(self.saved_backs) == 0:
                return
            need_to_run = (self.saved_backs[0][0],0)
        else:
            need_to_run = self.to_receive[0]
        i = 0
        while i < len(self.saved_backs):
            if self.saved_backs[i][0] == need_to_run[0]:
                data = self.saved_backs[i][1]
                del self.saved_backs[i]
                if len(self.to_receive) > 0:
                    del self.to_receive[0]
                bid = int.from_bytes(data[1:5],byteorder="big")
                frm = int.from_bytes(data[5:7],byteorder="big")
                originator = int.from_bytes(data[7:9],byteorder="big")
                
                nxt = self.send_receives.get(bid)

                if str(originator) == self.peer.pub_key:
                    nxt = -1
                    del self.send_receives[bid]
                else:
                    del self.send_receives[bid]
                self.queue_out.put(Backward(bid, frm, nxt, 0, 0, 0, originator, data[15:]), True)
                # Notify the next one:
                if nxt != originator:
                    await self.notify_next(nxt,data)
                break
            i += 1
        



    
    
    
    @bindto("get_peer")
    def _lower_get_peer(self, node_id)->Peer:
        return None
    
    @bindto("find_peer")
    async def _lower_find_peer(self, id: bytes) -> Peer:
        return None

    async def notify_next(self,nxt,data,quater = 1):
        if nxt == -1:
            return
        msg = bytearray([0])
        msg += data[1:5]
        msg += struct.pack(">f",time()/1000 + self.compute_time/quater)
        # with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
        #     log.write(f"Notifying previous {nxt}\n")
        p = await self._lower_find_peer(SHA256(str(nxt)))
        await self.send_datagram(msg,p.addr)
        # with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
        #     log.write(f"Notified previous {nxt}\n")
        return

    async def start_iteration(self):
        await asyncio.sleep(1)
        self.mb_send = 0
        
        for b in range(self.memory):
            
            if self.mb_send == self.MB_SEND_COUNT and self.memory == self.MAX_MEM:
                            
                await self.announce_end()
                return
            if self.mb_send == self.MB_SEND_COUNT:
                
                break
            
            tag = self.dp_order*self.MB_SEND_COUNT + self.mb_send
            with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                log.write(f"QUEUEIN MB {tag}\n")
            nxt = self.communication(tag,self.peer.pub_key)
            if nxt == None:
                
                sleep(1)
                await self.announce_end()
                return
            self.memory -= 1
            self.mb_send += 1
            

            self.queue_out.put(Start(tag,nxt,int(self.peer.pub_key)), True)

    async def start(self, p: Peer):
        await super().start(p)
        
        with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
            log.write(f"===={self.peer.pub_key} {self.stage} STARTING===\n")
        
        loop = asyncio.get_event_loop() 
        self.queue_reader = loop.create_task(self.read_from_queue())
        if self.stage == 0:
            await asyncio.sleep(2)
            loop.create_task(self.start_iteration())
        

    async def announce_end(self):
        self.mb_send = 0
        self.memory = self.MAX_MEM
        self.send_receives.clear()
        self.deferred.clear()
        self.queue_out.put(Aggregate(0), True)
        msg = bytearray()
        msg += PPProtocl_CA.AGGREGATE_FLAG.to_bytes(1,byteorder="big")
        for p in range(self.world_size):
            if str(p) == self.peer.pub_key:
                continue
            p = await self._lower_find_peer(SHA256(str(p)))
                                
            await self.send_datagram(msg, p.addr)
            

          

    async def read_from_queue(self):
        while self.started:
            while self.queue_in.empty() and self.started:
                await asyncio.sleep(0.1)
            if not self.started:
                with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                    log.write(f"====CLOSING???===\n")
                return
            task = self.queue_in.get(True)
            try:
                if isinstance(task, Forward):
                    self.processed.append(task.tag)
                    msg = bytearray()
                    msg += PPProtocl_CA.FORWARD_FLAG.to_bytes(1,byteorder="big")
                    msg += task.tag.to_bytes(4,byteorder="big")
                    msg += int(self.peer.pub_key).to_bytes(2,byteorder="big")
                    msg += task.originator.to_bytes(2,byteorder="big")
                    msg += task.B.to_bytes(2,byteorder="big")
                    msg += task.T.to_bytes(2,byteorder="big")
                    msg += task.C.to_bytes(2,byteorder="big")
                    msg += task.data
                    sndto = str(task.to)
                    # with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                    #     log.write(f"Will send to {sndto} mb {task.tag}\n")
                    p = await self._lower_find_peer(SHA256(sndto))
                    # with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                    #     log.write(f"FOUND {sndto}\n")
                    loop = asyncio.get_event_loop()
                    loop.create_task(self.send_stream(p.id_node,msg))
                     
                    continue
                elif isinstance(task, Loss):
                    # with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                    #     log.write(f"Sending loss grad to {task.to}\n")
                    msg = bytearray()
                    msg += PPProtocl_CA.BACK_FLAG.to_bytes(1,byteorder="big")
                    msg += task.tag.to_bytes(4,byteorder="big")
                    msg += int(self.peer.pub_key).to_bytes(2,byteorder="big")
                    msg += task.originator.to_bytes(2,byteorder="big")
                    msg += task.B.to_bytes(2,byteorder="big")
                    msg += task.T.to_bytes(2,byteorder="big")
                    msg += task.C.to_bytes(2,byteorder="big")
                    msg += task.data
                    sndto = str(task.to)
                    p = await self._lower_find_peer(SHA256(sndto))
                    loop = asyncio.get_event_loop()
                    loop.create_task(self.send_stream(p.id_node,msg))
                    # await 
                elif isinstance(task, Backward):
                    self.memory += 1
                    if self.stage != 0:
                        msg = bytearray()
                        msg += PPProtocl_CA.BACK_FLAG.to_bytes(1,byteorder="big")
                        msg += task.tag.to_bytes(4,byteorder="big")
                        msg += int(self.peer.pub_key).to_bytes(2,byteorder="big")
                        msg += task.originator.to_bytes(2,byteorder="big")
                        msg += task.B.to_bytes(2,byteorder="big")
                        msg += task.T.to_bytes(2,byteorder="big")
                        msg += task.C.to_bytes(2,byteorder="big")
                        msg += task.data
                        sndto = str(task.to)
                        p = await self._lower_find_peer(SHA256(sndto))
                        loop = asyncio.get_event_loop()
                        loop.create_task(self.send_stream(p.id_node,msg))
                        # await 
                        loop.create_task(self.start_back())
                        # await 
                        
                        if len(self.deferred) > 0:
                            loop = asyncio.get_event_loop()
                            tg = self.deferred.pop()
                            loop.call_later(0.1,self.process_data, tg[0],tg[1],tg[2])
                            # return
                    else:
                        with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                            log.write(f"CAN WE START BACK {self.memory} {self.mb_send} {self.MB_SEND_COUNT} {self.MAX_MEM}\n")
                        if self.mb_send == self.MB_SEND_COUNT and self.memory == self.MAX_MEM:
                            
                            await self.announce_end()
                        elif self.mb_send > self.MB_SEND_COUNT:
                            raise Exception(f"Too many microbatches have been sent? {self.memory} {self.MAX_MEM} {self.mb_send} {self.MB_SEND_COUNT}")
                        
                        continue

                    continue
                elif isinstance(task, Aggregate):
                    self.iteration += 1
                    if self.stage != 0:
                        continue
                    self.mb_send = 0
                    
                    # s\elf.memory = self./
                    loop = asyncio.get_event_loop() 
                    loop.create_task(self.start_iteration())
            except Exception as e:
                with open(f'log{self.peer.pub_key}.txt', 'a') as f:
                    f.write(str(e))
                    f.write("!!!!!!!!!!!!!!!\n")
                    f.write(format_exc())
                    


    @bindfrom("connected_callback")
    def peer_connected(self, nodeid, peer: Peer):
        
        with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                log.write(f"CONNECTED WITH {peer.pub_key}\n")
       
        return self.connected_callback(nodeid,peer)
    

 
    def process_datagram(self, addr: tuple[str, int], data: bytes):
        
        
        if data[0] == PPProtocl_CA.AGGREGATE_FLAG:
            if self.stage == 0:
                return
            self.received_aggregates += 1
            with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                log.write(f"AGGREGATE RECEIVED\n")
            if self.received_aggregates >= self.datanodes:
                with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                    log.write(f"AGGREGATING\n")
                self.received_aggregates = 0
                self.forwards_collected = 0
                self.processed.clear()
                self.send_receives.clear()
                self.deferred.clear()
                self.memory = self.MAX_MEM
                self.queue_out.put(Aggregate(0), True)
        else:
            
        
            mb_id = int.from_bytes(data[1:5],byteorder="big")
            # with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
            #     log.write(f"Notified about {mb_id}\n")
            tm = struct.unpack(">f",data[5:])[0]
            p_len = self.p_len(mb_id,int(self.peer.pub_key))
            self.to_receive.append([mb_id,tm,p_len])
            self.run_back_scheduler()
         
        return

   
    @bindto("get_peer")
    def _lower_get_peer(self, node_id)->Peer:
        return None
    @bindto("find_peer")
    async def _lower_find_peer(self, id: bytes) -> Peer:
        return None
    
    async def stop(self):
        
        await super().stop()
        
        self.queue_in.close()
        self.queue_out.close()

    @bindfrom("stream_callback")
    def process_data(self, data:bytes, nodeid, addr):
        if data[0] == PPProtocl_CA.FORWARD_FLAG:
            bid = int.from_bytes(data[1:5],byteorder="big")
            frm = int.from_bytes(data[5:7],byteorder="big")
            self.send_receives[bid] = frm
            self.forwards_collected += 1
            originator = int.from_bytes(data[7:9],byteorder="big")
            with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                log.write(f"Will receive from {frm} mb {bid} originator {originator} {time()}\n")
            B = int.from_bytes(data[9:11],byteorder="big")
            T = int.from_bytes(data[11:13],byteorder="big")
            C = int.from_bytes(data[13:15],byteorder="big")
            if self.memory == 0 and self.peer.pub_key != str(originator):
                self.deferred.append((data,nodeid,addr))
                return
            elif self.peer.pub_key != str(originator):
                self.memory -= 1
            nxt = self.communication(bid,self.peer.pub_key)
            if nxt == None and self.peer.pub_key != str(originator):
                nxt = originator
            elif self.peer.pub_key == str(originator):
                with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                    log.write(f"NEED TO COMPUTE LOSS FROM {frm} mb {bid}\n")
                
                loop = asyncio.get_event_loop()
                loop.create_task(self.notify_next(frm,data[0:15],4))
                self.queue_out.put(Loss(bid, frm, frm, B, T, C, originator, data[15:]), True)
                return

            
            
            self.queue_out.put(Forward(bid, frm, nxt, B, T, C, originator, data[15:]), True)

            return
        elif data[0] == PPProtocl_CA.BACK_FLAG:
            bid = int.from_bytes(data[1:5],byteorder="big")
            frm = int.from_bytes(data[5:7],byteorder="big")
            originator = int.from_bytes(data[7:9],byteorder="big")
            with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                log.write(f"Will receive backward from {frm} mb {bid} originator {originator} {self.send_receives}\n")
            if str(originator) == self.peer.pub_key:

                nxt = self.send_receives.get(bid)

                if str(originator) == self.peer.pub_key:
                    self.queue_out.put(Backward(bid, frm, nxt, 0, 0, 0, originator, data[15:]), True)
                    if self.mb_send < self.MB_SEND_COUNT:
                            
                            self.memory -= 1
                            tag = bid
                            self.mb_send += 1
                            nxt = self.communication(tag,self.peer.pub_key)
                            self.queue_out.put(Start(tag,nxt,int(self.peer.pub_key)), True)
                    elif self.mb_send == self.MB_SEND_COUNT and self.memory == self.MAX_MEM:
                            loop = asyncio.get_event_loop()
                            loop.create_task(self.announce_end())
                    nxt = -1
                    del self.send_receives[bid]
                else:
                    del self.send_receives[bid]
                    self.queue_out.put(Backward(bid, frm, nxt, 0, 0, 0, originator, data[15:]), True)
                
                             
                            
            else:
                bid = int.from_bytes(data[1:5],byteorder="big")
                self.saved_backs.append((bid,data))
                loop = asyncio.get_event_loop()
                loop.create_task(self.start_back())
            
            
            

        
       
    async def send_stream(self, node_id, data):
        
        await self._lower_find_peer(bytes(node_id))
        p = self._lower_get_peer(node_id)
        await self._lower_open_connection(p.addr[0], p.tcp, p.id_node, port_listen = 0)
        loop = asyncio.get_event_loop()
        loop.create_task(self._lower_send_stream(node_id, data))
         
        return
    
    @bindto("open_connection")
    async def _lower_open_connection(self, remote_ip, remote_port, node_id: bytes):
        return
    @bindto("send_stream")
    async def _lower_send_stream(self, node_id, data):
        return
                
    def get_lowest_stream(self):
        submodule = self.submodule
        while submodule != None and not hasattr(submodule, "get_lowest_stream") and hasattr(submodule, "submodule") :
            submodule = submodule.submodule
        if submodule != None and hasattr(submodule, "get_lowest_stream"):
            ret = submodule.get_lowest_stream()
            if ret == None:
                return self
            else:
                return ret
        else:
            
            return self