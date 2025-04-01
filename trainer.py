import random
import time
from deccom.protocols.peerdiscovery.kademliadiscovery import KademliaDiscovery
from sys import argv
import asyncio
from deccom.cryptofuncs.hash import SHA256
from deccom.nodes import StreamNode, Node
from deccom.protocols.defaultprotocol import DefaultProtocol
from deccom.peers import Peer
from deccom.protocols.streamprotocol import StreamProtocol

from multiprocessing import Lock, Process, Queue, current_process
import json
from schedulers.communication_costs import delay_map
from deccom.protocols.delayprotocol import DelayProtocol
from pprint import pprint

from schedulers.communication_costs import *

# LLaMa 1.5B parameters:
seq_l = 4096
n_layers = 1 # this is per device... we did 6 stages, so 4 layers per device. Adjust this up to your needs
batch_size = int(argv[4])
device = argv[5]
dmodel = 2048
num_heads = 16

# COMMENT OUT:
# seq_l = 128
# n_layers = 2
# batch_size = 4
# dmodel = 288
# num_heads = 2

if __name__ == '__main__':
    curr_id = int(argv[1])
    setting = argv[2]

    communication_distribution = argv[3]
    loop = asyncio.new_event_loop()
    with open(f"schedule.json", 'r') as file:
        config = json.load(file)
    def delay_map(currid,otherid):
        p1 = config["locations"][int(currid)]
        p2 = config["locations"][int(otherid)]
        if DELAY_BANDWIDTHS.get(p1+"-"+p2) != None:
            ret = DELAY_BANDWIDTHS.get(p1+"-"+p2)
        elif DELAY_BANDWIDTHS.get(p2+"-"+p1) != None:
            ret = DELAY_BANDWIDTHS.get(p2+"-"+p1)
        else:
            ret = (10,2.00)

        return (ret[0] - 0.1, ret[1]) # we experience about 0.1 communication time, so remove it from the simulation
    loc = config["locations"][curr_id]
    world = len(config["locations"])
    cost_map = [[0 for _ in range(world)] for _ in range(world)]
    for y in range(world):
        for x in range(world):
            cost_map[y][x] = delay_map(y,x)[1]
    # print(loc)
    compute_time = get_computations(communication_distribution)[loc]*n_layers*batch_size
    world_size = 0
    own_stage = -1
    rank_order = 0
    partitions = config["partitions"]
    memory = config["memory"]
    send_mbs = 0
    if setting == "baseline" or setting == "zbh1":
        send_mbs = int(config["baseline-sends"])
        from communications.pp_protocol import PPProtocl as PPProtocl
    elif setting == "random" or setting == "non-ca-partial" or setting == "ca-partial":
        send_mbs = config["ours-sends"]
        from communications.pp_protocol import PPProtocl as PPProtocl
    else:
        # TODO: experimental idea... see if it works
        send_mbs = config["ours-sends"]
        from communications.pp_protocol_ca import PPProtocl_CA as PPProtocl

    for idx, v in enumerate(partitions):
        if curr_id in v:
            assert own_stage == -1
            own_stage = idx
            for idx2, v2 in enumerate(v):
                if v2 == curr_id:
                    rank_order = idx2
                    break
        world_size += len(v)
    if setting == "zbh1":
        from communications.llm_subp_zbh1 import run_p
    else:
        from communications.llm_subp import run_p
    assert own_stage != -1
    def p_len(bid,ndkey):
        b1 = bid % config["ours-sends"]
        b2 = bid // config["ours-sends"]
        b1 = b1 % config["memory"]
        b_path = config["ca-paths"][str(b2*config["memory"] + b1)]
        ln = 1
        for k,v in b_path.items():
            if v == ndkey:
                return ln
            ln += 1
        return ln
    def commfunc(bid, ndkey):
        if setting == "baseline" or setting == "zbh1":
            if own_stage == len(partitions) - 1:
                return None
            if len(partitions[own_stage + 1]) <= rank_order:
                return None
            # with open(f"log_stats_proj_2_{curr_id}.txt", "a") as log:
            #     log.write(f"Paritions {partitions[own_stage + 1]}, {rank_order}, {own_stage}\n")
            return partitions[own_stage + 1][rank_order]
        
        
        elif setting == "random":
            # ca-paths
            b1 = bid % config["ours-sends"]
            b2 = bid // config["ours-sends"]
            b1 = b1 % config["memory"]
            b_path = config["random-paths"][str(b2*config["memory"] + b1)]
            if ndkey in b_path:
                return b_path[ndkey]
            else:
                return None
        elif setting == "non-ca-partial":
            b1 = bid % config["ours-sends"]
            b2 = bid // config["ours-sends"]
            b1 = b1 % config["memory"]
            b_path = config["non-ca-paths"][str(b2*config["memory"] + b1)]
            if ndkey in b_path:
                return b_path[ndkey]
            else:
                return None
        else:
            # ca-paths
            b1 = bid % config["ours-sends"]
            b2 = bid // config["ours-sends"]
            b1 = b1 % config["memory"]
            b_path = config["ca-paths"][str(b2)]
            if ndkey in b_path:
                return b_path[ndkey]
            else:
                return None
            
    while True:
        my_peer  = Peer(None, pub_key=str(curr_id))
        
        port = None

        protocol = DefaultProtocol()
        gossip = KademliaDiscovery([],interval=30, always_split = True)
        gossip.set_lower(protocol)
        stream = StreamProtocol(False)
        stream.set_lower(gossip)
        delayer = DelayProtocol(delay_map,True)
        delayer.set_lower(stream)
        n = Peer(("127.0.0.1", 10015))
        if curr_id != 0:
            gossip.bootstrap_peers.append(n)
            time.sleep(1)
        



        queue_in = Queue(1024)
        queue_out = Queue(1024)
        
        if curr_id > 5:
            device = "cuda:1"
        if curr_id > 10:
            device = "cuda:2"
        if curr_id > 15:
            device = "cuda:3"
        if curr_id > 21:
            device = "cuda:4"
        if curr_id > 26:
            device = "cuda:5"
        if curr_id > 32:
            device = "cuda:6"
        if curr_id > 37:
            device = "cuda:7"
        subprocess = Process(target=run_p,args=(n.addr[0],partitions,queue_out,queue_in,curr_id,own_stage,seq_l,n_layers,batch_size,dmodel,num_heads,memory,compute_time,send_mbs,cost_map, device)) 
        trainingp = PPProtocl(world_size, own_stage, commfunc, None, len(partitions[0]), memory, queue_in, queue_out, subprocess, MB_SEND_COUNT=send_mbs, dp_order=rank_order)
        trainingp.set_lower(delayer)
        if setting == "ca-partial-new":
            trainingp.p_len = p_len
            trainingp.compute_time = compute_time
        subprocess.start()
        
        me = StreamNode(my_peer , trainingp,ip_addr="0.0.0.0", port = 10015 if curr_id == 0 else port)
        # print( "TCP", me.tcp_port)

        
        print("run...")
        
        loop.run_until_complete(me.listen())
        loop.run_forever()
        
