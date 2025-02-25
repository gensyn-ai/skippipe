# SkipPipe: Partial and Reordered Pipelining Framework for Training LLMs in Heterogeneous Networks
This repository contains the code and instructions to replicate experiments of the paper titled ["SkipPipe: Partial and Reordered Pipelining Framework for Training LLMs in Heterogeneous Networks"](link).



## Requirements:

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
  eprint={ },
  archivePrefix={arXiv},
  primaryClass={cs.DC}
}
```
