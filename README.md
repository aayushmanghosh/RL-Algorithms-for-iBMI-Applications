# RL-Algorithms-for-iBMI-Applications

RL-Algorithms-for-iBMI-Applications is a code repository that contains various algorithms that facilitates the application of five popular online RL Algorithms (Banditron, Banditron-RP, Deep Q-Learning, AGREL, and HRL) in the context of intention decoders for iBMI systems. 

## About this repo: 
This repo hosts the MATLAB source codes that directly acts on in-house recorded dataset (a small portion of it is [publicly available](https://osf.io/dce96/)) to generate feature_sets (included in this github repository under the directory ..\datasets\derived-RL-expt), which are used as an input to the RL-algorithms to record the decoding accuracy. The decoding accuracy is generated as an output of a multiclass classification experiment (4 classes), where each class denotes separate directions. Detailed information regarding the used datasets are outlined in this [paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0165773).  
