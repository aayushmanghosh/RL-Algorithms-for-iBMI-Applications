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







