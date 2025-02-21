from dataclasses import dataclass
from multiprocessing import Lock, Process, Queue, current_process
from torch import manual_seed
from torch.distributed import recv, isend, irecv, send
from torch import Tensor, zeros, save
from simplellm.tokenizers import SPTokenizer
from simplellm.llama import LLamaFirstStage, LLamaStage
from .dp_group import DP_Group, initialise_communication
from .dp_optimizer import DP_optim
from torch import cuda, no_grad
import traceback
import torch
from contextlib import redirect_stdout
from simplellm.dataloaders import Wikipedia_Dataset, TinyStories
import torch.nn.functional as F
from time import time, sleep
from simplellm.losses import causalLLMLoss
import pickle
# Messages Exchanged by the processes
@dataclass
class Forward:
    tag: int
    frm: int
    to: int
    B: int
    T: int
    C: int
    originator: int
    data: Tensor
@dataclass
class Backward:
    tag: int
    frm: int
    to: int
    B: int
    T: int
    C: int
    originator: int
    data: Tensor

@dataclass
class Start:
    tag: int
    to: int
    originator: int
@dataclass
class Deferred:
    tag: int
@dataclass
class Loss:
    tag: int
    frm: int
    to: int
    B: int
    T: int
    C: int
    originator: int
    data: Tensor



@dataclass
class Aggregate:
    epoch: int

def run_p(main_addr, partitions, queue_in: Queue, queue_out: Queue, node_id: int = 0, stage: int = 0, seq_l: int = 256, n_layers = 4, 
                    batch_size = 8, dmodel = 256, num_heads = 16, memory = 3, process_time = 2, mb_count = 12, cost_map = [],
                    device = "cuda"):
    manual_seed(0)
    world_size = 0
    for v in partitions:
        world_size += len(v)
    group = initialise_communication(partitions,node_id, main_addr, world_size,cost_map)
    
    if stage == 0:
        tkns = SPTokenizer()
        ts = Wikipedia_Dataset(tkns,batch_size = batch_size, seq_l=seq_l,skip=group.in_group*8_000)
        net = LLamaFirstStage(tkns.vocab_size, dmodel, num_heads, n_layers, ctx_size= seq_l,device=device)
        
        optimizer = DP_optim(6e-4, net, group, device)
        with open(f'log{node_id}.txt', 'a') as file, redirect_stdout(file):
            loc =  SubP(queue_in,queue_out,net,optimizer,node_id,stage,ts,device=device, mb_count=mb_count, memory = memory,process_time=process_time)
            loc.start()
    else:
        net = LLamaStage(ctx_size=seq_l, dmodel=dmodel,num_heads=num_heads,n_layers=n_layers,device=device)
        
        optimizer = DP_optim(6e-4, net, group, device)
        
        loc =  SubP(queue_in,queue_out,net,optimizer,node_id,stage,None,device=device,  mb_count=mb_count, memory = memory,process_time=process_time)
        loc.start()


class SubP(object):
    def __init__(self,queue_in: Queue, queue_out: Queue, net, optimizer, node_id = 0, stage = 0, ds = None,
                    device = "cuda", mb_count = 12, process_time = 2, memory = 3) -> None:
        self.net = net
        self.process_time = process_time
        self.memory = memory
        self.MAX_MEM = memory
        self.device = device
        self.queue_in: Queue = queue_in
        self.queue_out: Queue = queue_out
        self.optimizer = optimizer
        
        self.node_id = node_id
        self.buffer_in = {}
        self.buffer_out = {}
        self.receives = []
        self.iteration = 0
        self.mb_count = mb_count
        self.started = True
        self.deferred = {}
        self.mbs = []
        self.epoch = 0
        
        if stage == 0:
            self.ds = ds
            self.dl = iter(ds)
            
            self.target = {}
        
        
        


    def start(self):
        try:
            while self.started:
                
                while self.queue_in.empty() and self.started:
                    continue
                if not self.started:
                    break
                
                task = self.queue_in.get(True)
                if isinstance(task, Start):
                    tm1 = time()
                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        # log.write(f"Processing forward to {task.to} {time()}\n")
                        log.write(f"=======NEW MB:======== {time()}\n")
                    
                    if len(self.mbs) == 0:
                        while len(self.mbs) < 45:
                            self.mbs.append(next(self.dl))
                    x = self.mbs.pop(0)
                    self.target[task.tag] = x.detach().clone()
                    with no_grad():
                        x = x.to(self.device)

                    self.buffer_in[task.tag] = x
                    x = self.net.embed(x)
                    x.retain_grad()
                    self.buffer_out[task.tag] = x
                    ret = pickle.dumps(x)
                    self.memory -= 1
                    tm2 = time()
                    if tm2 - tm1 < (self.process_time):
                        sleep((self.process_time) - (tm2 - tm1)) # due to us simulating multiple devices on the same gpu, we need to introduce these tricks to 
                        # reasonably simulate execution of separate nodes
                    

                    self.queue_out.put(Forward(task.tag, self.node_id, task.to, x.shape[0], x.shape[1], x.shape[2], task.originator, ret), True)
                    
                    
                    continue
                elif isinstance(task, Loss):
                    tm1 = time()
                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        log.write(f"LOSSS {time()}\n")
                    
                    
                    x = pickle.loads(task.data)
                    with no_grad():
                        x = x.to(self.device)
                    x.requires_grad = True
                    x.retain_grad()
                    y = self.target[task.tag].to(self.device)
                    ret = self.net.forward_end(x)

                    loss = causalLLMLoss(ret,y,vocab_size=self.ds.tokenizer.vocab_size)
                    loss_report = loss.item()
                    loss = loss / self.mb_count
                    tm2 = time()
                    if tm2 - tm1 < (self.process_time)/4:
                        sleep((self.process_time)/4 - (tm2 - tm1))
                    ret = x.grad
                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        log.write(f"LOSS:{loss_report} {tm2-tm1}\n") #
                    # ret = ret.to("cpu")
                    
                    self.queue_out.put(Loss(task.tag, task.frm, task.to, x.grad.shape[0], x.grad.shape[1], x.grad.shape[2], task.originator, pickle.dumps(ret)), True)
                    
                elif isinstance(task, Forward):
                    
                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        log.write(f"Processing forward to {task.to} {time()}\n")
                    tm1 = time()
                    x = pickle.loads(task.data)
                    self.memory -= 1
                    with no_grad():
                        x = x.to(self.device)
                    x.requires_grad = True
                    x.retain_grad()

                    self.buffer_in[task.tag] = x
                    
                    
                    x = self.net(x)
                    x.retain_grad()
                    self.buffer_out[task.tag] = x

                    ret = pickle.dumps(x)
                    tm2 = time()
                    if tm2 - tm1 < (self.process_time):
                        sleep(self.process_time  - (tm2 - tm1))

                    self.queue_out.put(Forward(task.tag, task.frm, task.to, x.shape[0], x.shape[1], x.shape[2], task.originator, ret), True)
                    
                    continue
                    
                elif isinstance(task, Backward):
                    
                    tm1 = time()
                    output = pickle.loads(task.data)
                    
                    
                    with no_grad():
                        output = output.to(self.device)
                        
                    inp_batch = self.buffer_out[task.tag].to(self.device)
                    
                    inp_batch.backward(output)
                    tm2 = time()
                    
                    self.memory += 1
                    
                    if task.to != -1:
                        ret = self.buffer_in[task.tag].grad
                        if tm2 - tm1 < 1.5*(self.process_time):
                            sleep(1.5*(self.process_time)  - (tm2 - tm1))

                        self.queue_out.put(Backward(task.tag, task.frm, task.to, ret.shape[0], ret.shape[1], ret.shape[2], task.originator, pickle.dumps(ret)),True)
                        
                    else:
                        if tm2 - tm1 < 1.5*(self.process_time):
                            sleep(1.5*(self.process_time) - (tm2 - tm1))

                        self.queue_out.put(Backward(task.tag, task.frm, task.to, 0, 0, 0, task.originator, None),True)

                    del self.buffer_in[task.tag]
                    del self.buffer_out[task.tag] 
                    cuda.empty_cache()
                    

                    
                
                elif isinstance(task, Aggregate):
                    assert self.memory == self.MAX_MEM
                    with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                        log.write(f"===AGGEGATING==== {time()}\n")
                    self.buffer_in.clear()
                    self.buffer_out.clear()
                    cuda.empty_cache()
                    self.iteration += 1

                    # update params
                    self.optimizer.step() # this also syncs across stage
                    
                    
                    if self.iteration % 2000 == 0 and self.optimizer.dp_group.in_group == 0::
                       save(self.net.state_dict(), f"{self.optimizer.dp_group.partition}.pth") 
                    cuda.empty_cache()
                    self.queue_out.put(Aggregate(0), True)
                    
                    if len(self.mbs) > 0:
                        while len(self.mbs) < 45:
                            self.mbs.append(next(self.dl)) # sometimes huggingface would delay streaming of ds, so pre-get it
                    self.optimizer.sync()
        except Exception:
            with open(f"log_stats_proj_2_{self.node_id}.txt", "a") as log:
                log.write(f"{traceback.format_exc()}\n")
            
            exit()