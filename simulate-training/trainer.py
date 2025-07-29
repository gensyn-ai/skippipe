from simplellm.llama import SwapLLama, LLama
from simplellm.tokenizers import SPTokenizer
from simplellm.dataloaders import RedPyjama, PretrainDataset
from sys import argv
import torch.distributed as dist
from torch import save, cuda, zeros_like, cat, mean, std
import torch
import traceback
import os
from simplellm.utils import State
import random
from torch.optim import Adam
import json
from simplellm.losses import causalLLMLoss, perplexityLoss
random.seed(42)
State.set_seed(42)
torch.manual_seed(3407)
rank = int(argv[1])
world_size = int(argv[2])
skip = int(argv[3])

device = f"cuda:{rank}"
dim = 1024
kv_heads = 16
layers = 24
stages = 6
layers_per_stage = layers // stages
ctx_size = 1024
lr = 3e-4
mb_c = 6
num_warmup_steps = 500

tokenizer = SPTokenizer()
padding_idx = tokenizer.eos_id
train_ds = RedPyjama(tokenizer, batch_size=8, skip = 100, group="default", seq_l=ctx_size)
val_ds = RedPyjama(tokenizer, batch_size=8, skip = 0, seq_l=ctx_size)
net = LLama(SwapLLama,tokenizer.vocab_size, dmodel=dim, num_heads=kv_heads, n_layers=layers, ctx_size=ctx_size, padding_idx=padding_idx, device=device)
with open("2_communication_8_samples_llama_500M","r") as fd:
    config = json.load(fd)
paths = config["ca-paths"]
partitions = config["partitions"]
tmp = {}
for idx, p in partitions:
    for nd in p:
        tmp[p] = idx
partitions = tmp
sizes = []
len_sizes = []
for param in net.parameters():
    sizes.append(param.shape)
    len_sizes.append(len(param.view(-1)))

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group("nccl", rank=rank, world_size=world_size)
tmp = []
for param in net.parameters():
    if param.data == None:
        tmp.append(torch.zeros_like(param,device=device).view(-1))                      
        continue
    tmp.append(param.data.view(-1))

tmp = cat(tmp)
dist.all_reduce(tmp, op = dist.ReduceOp.AVG)
tmp = torch.split(tmp, len_sizes)
# Sync model across devices...
for pi, param in enumerate(net.parameters()):
    param.data = tmp[pi].view(sizes[pi]).to(device)

optimizer = Adam(net.parameters(),lr = lr, betas=(0.9, 0.999), weight_decay=0)
train_dl = iter(train_ds)
for itr in range(25_000):
    optimizer.zero_grad()
    if itr % 100 == 0 and rank == 0:
        net.eval()
        loss_hist = []
        with torch.no_grad():
            order = list(range(layers))
            val_dl = iter(val_ds)
            for _ in range(100):
                x = next(val_dl)
                target = x.detach().clone()
                x = net(x, order = order)
                loss_hist.append(perplexityLoss(x,target).item())
            print(itr, "VALIDATION LOSS", sum(loss_hist)/len(loss_hist))
        save(net.state_dict(), "mdl.pth")
        save(optimizer.state_dict(), "optim.pth")
        net.train()
    
    loss_hist = 0
    for mb in range(mb_c):
        for k in range(world_size):
            try:
                if k == rank:
                    x = next(train_dl)
                else:
                    next(train_dl)
            except StopIteration:
                train_dl = iter(train_ds)
                if k == rank:
                    x = next(train_dl)
                else:
                    next(train_dl)
        target = x.detach().clone()
        if skip == 0:
            order = list(range(layers))
            output = net(x, order = order)
        else:
            order = [kl for kl in range(layers_per_stage)]
            mb = paths[mb // 2 + rank * 3]
            for v in mb.values():
                order += list(range(layers_per_stage * partitions[v], layers_per_stage * (1 + partitions[v])))
            print(mb)
            print(order)
            
            output = net(x, order = order)


        loss = causalLLMLoss(output, x, tokenizer.vocab_size) / mb_c
        loss_hist += loss.item()
        loss.backward()
    print(itr,"TRAINING LOSS", loss_hist)
    dist.barrier()
    tmp = []
    for param in net.parameters():
        if param.grad == None:
            tmp.append(torch.zeros_like(param,device=device).view(-1))                      
            continue
        tmp.append(param.grad.view(-1))

    tmp = cat(tmp)
    dist.all_reduce(tmp, op = dist.ReduceOp.AVG)
    tmp = torch.split(tmp, len_sizes)
    # Sync model across devices...
    for pi, param in enumerate(net.parameters()):
        param.grad = tmp[pi].view(sizes[pi]).to(device)
    optimizer.step()
    del tmp
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    











