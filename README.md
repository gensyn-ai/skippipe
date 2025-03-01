# SkipPipe: Partial and Reordered Pipelining Framework for Training LLMs in Heterogeneous Networks
This repository contains the code and instructions to replicate experiments of the paper titled ["SkipPipe: Partial and Reordered Pipelining Framework for Training LLMs in Heterogeneous Networks"](https://arxiv.org/abs/2502.19913).

SkipPipe introduces a fault tolerant, pipeline parallel method that skips and reorders stages dynamically to optimize training within decentralized environments. SkipPipe shows a 55% reduction in training time compared to standard pipeline methods within these environments, with no degradation in convergence.

It is also highly fault tolerant - demonstrating robustness up to 50% node failure rate with only 7% perplexity loss at inference time (i.e. when half of the pipeline nodes for a single model are unavailable we only lose 7% perplexity running inference through the - now sparse - model).

Unlike existing data parallel methods, SkipPipe can accommodate large model training. Since it shards the model itself across nodes, rather than simply sharding the dataset, SkipPipe reduces the memory footprint on each individual node and removes the cap on model size - allowing models of theoretically infinite size to be built across distributed, and decentralised, infrastructure. 

![SkipPipe Figure](/assests/skippipe.png)
*An example of partial pipeline parallelism scheduling where each colored (solid or dashed) path represents a different microbatch. Each node in stage 0 sends out 2 microbatches, the first in solid, the second in dashed. Green backgrounds show the forward pass, while light orange - the backwards pass. For better visualization, the loss and deembedding computations are omitted. Arrows show the prioritisation of the microbatches from forward to backward pass within the same node.")*

## Requirements

This code uses the following two repositories:

- [simplellm](https://github.com/NikolayBlagoev/simplellm) - for construction of the models, loading datasets, tokenizers, etc.

- [DecCom](https://github.com/NikolayBlagoev/DecCom-Python) - for communication between devices

You can install both by cloning the repo and doing ```pip install .``` or by running the [setup.sh](/setup.sh) provided here.

Additionally, you need to install the requirements in [requirements.txt](/requirements.txt) with ```pip install -r requirements.txt```


## Making a scheduler

Schedulers are made with [create_schedule.py](create_schedule.py). Modify the respective hyper parameters of the algorithm (lines 16 to 42). Depending on your CPU and setting this may take a bit...


## Training

Start training with

```
./run.sh [FIRST DEVICE] [LAST DEVICE] [SETTING] [SAMPLES PER MICROBATCH]
```

Which will start all nodes from FIRST DEVICE to LAST DEVICE on this machine with a given SETTING (*random* for DT-FM Skip, *ca-partial* for SkipPipe with TC2, *non-ca-partial* for SkipPipe without TC2, or *baseline* for DT-FM). 


## Publication

```bibtex
@article{blagoev2025skippipe,
  title={SkipPipe: Partial and Reordered Pipelining Framework for Training LLMs in Heterogeneous Networks}, 
  author={Blagoev, Nikolay and Chen, Lydia Y and Ersoy, O\u{g}uzhan},
  year={2025},
  eprint={2502.19913},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2502.19913},
}
```
