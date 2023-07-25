# RL-Algorithms-for-iBMI-Applications

RL-Algorithms-for-iBMI-Applications is a code repository that contains various algorithms that facilitates the application of five popular online RL Algorithms (Banditron, Banditron-RP, Deep Q-Learning, AGREL, and HRL) in the context of intention decoders for iBMI systems. 

## About this repo: 
This repo hosts the MATLAB source codes that directly acts on in-house recorded dataset (a small portion of it is [publicly available](https://osf.io/dce96/)) to generate feature_sets (included in this github repository under the directory '..\datasets\derived-RL-expt'), which are used as an input to the RL-algorithms to record the decoding accuracy. The decoding accuracy is generated as an output of a multiclass classification experiment (4 classes), where each class denotes separate directions. Detailed information regarding the used datasets are outlined in this [paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0165773).  

## Introduction:

Intra-cortical Brain Machine Interfaces (iBMIs) with wireless capability could scale the number of recording channels by integrating an intention decoder to reduce data rates. However, the need for frequent retraining due to neural signal non-stationarity is a big impediment. Hence we present an alternate neuromorphic paradigm of online reinforcement learning (RL) with a binary evaluative feedback in iBMIs to tackle this issue. This paradigm eliminates time-consuming calibration procedures. Instead, it relies on updating the model on a sequential sample-by-sample basis based on an instantaneous evaluative binary feedback signal. Such online learning is a hallmark of neuromorphic systems and is different from batch updates of weight in popular deep networks that is very resource consuming and incompatible with constraints of an implant. We show application of simple RL algorithms - Banditron in discrete-state iBMIs and compare it against previously reported state of the art RL algorithms -- Hebbian RL (HRL), Attention Gated RL (AGREL), deep Q-learning. Furthermore, we propose a non-linear variant of Banditron, Banditron-RP, which gives an average improvement of power consumption. A resistive RAM (RRAM) based implementation of the same is also simulated to show feasiblity of such hardware in future implants.

## Using the Code: 
- Clone this repository:
```bash
git clone https://github.com/aayushmanghosh/RL-Algorithms-for-iBMI-Applications
cd RL-Algorithms-for-iBMI-Applications
```
This code is stable using Python 3.9.4, Tensorflow 2.13.0
- To install all the dependencies using pip:
```bash
pip install -r requirements.txt
```

### Overview of this repository:
1) [Datasets](https://github.com/aayushmanghosh/RL-Algorithms-for-iBMI-Applications/tree/main/datasets) contains three separate folders: (a) derived-RL-expt, (b) original, (c) synthetic noise. Derived-RL-expt contains the input feature datasets to *BanditronRP-RRAM-data.py* and *RL-decoders-iBMI.py*. Original contains the publicly available dataset. Synthetic noise also contain the input feature dataset but here additional noise is added to the spike count. Refer to the original publication for more information.

2) [plots-and-figures](https://github.com/aayushmanghosh/RL-Algorithms-for-iBMI-Applications/tree/main/plots-and-figures) contains two separate folders: (a) functions (b) publication-specific-scripts. Functions stores the source code to generate publication specific graphs. Publication-specific-scripts stores the matlab scripts actually used to generate the graphs used in our paper with all the required edits.

3) [source-codes](https://github.com/aayushmanghosh/RL-Algorithms-for-iBMI-Applications/tree/main/source-codes) contains MATLAB file for generating the required derived-RL-expt and synthetic noise datasets. *'spike_count.m'* generates the derived-RL-expt dataset. It counts the number of neuron spikes occuring in a backward looking window. *'syn_dataset.m'* generates the synthetic noise datasets. It works based on the Isekevich model to generate the noisy dataset.

4) [BanditronRP-RRAM-data.py](https://github.com/aayushmanghosh/RL-Algorithms-for-iBMI-Applications/blob/main/BanditronRP-RRAM-data.py) lists the BanditronRP algorithm where the weights of the internal layer is modified based on the experimental data obtained from RRAM implementation. 

5) [RL-decoders-iBMI.py](https://github.com/aayushmanghosh/RL-Algorithms-for-iBMI-Applications/blob/main/RL-decoders-iBMI.py) stores all the RL-algorithms used as a decoder for iBMI applications. Refer to the original publication for more information.

### Note:
Various training modalities are also employed to generate excess data for the publication. Batch training, sequential training and group trainings are also performend. The codes for those training modalities are not included in this repository as they are widely available over the web.

### Citation:
A. Ghosh, S. Shaikh, P. S. V. Sun, C. Libedinsky, R. So, N. Lin, Z. Wang, A. Basu, "Low-complexity Reinforcement Learning Decoders for Autonomous, Scalable, Neuromorphic intra-cortical Brain Machine Interfaces," Neuromorphic Computing and Engineering (Under review)

Open an issue or mail me directly in case of any queries or suggestions.
