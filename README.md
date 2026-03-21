
# Adaptive Graph Integration for Cross-Domain Recommendation via Heterogeneous Graph Coordinators

> 📝 SIGIR 2025

## 🔬 Overview

In this project, we introduce HAGO, a novel framework with <u>**H**</u>eterogeneous <u>**A**</u>daptive <u>**G**</u>raph co<u>**O**</u>rdinators, which dynamically integrate multi-domain graphs into a cohesive structure by adaptively adjusting the connections between coordinators and multi-domain graph nodes, thereby enhancing beneficial inter-domain interactions while mitigating negative transfer effects. On this basis, we develop a universal pre-training framework that can integrate various self-supervised learning algorithms alongside HAGO to collaboratively learn high-quality node representations across multiple domains.

<img src=HAGO.png width=800 height=270 />

## 🌟 Environment Setup

### Prerequisites

The main prerequisites are listed below:
```
Python 3.8.19
torch==2.3.1
recbole==1.0.1
pygcl==0.1.2
dgl==2.4.0
```

And the entire dependencies can be set up by running:

```
pip install -r requirements.txt
```


### Datasets

Download the datasets via the following URL and move the datasets to ./dataset folder.

- [`Douban`](https://recbole.s3-accelerate.amazonaws.com/CrossDomain/Douban.zip) datasets;
- [`Amazon`](http://jmcauley.ucsd.edu/data/amazon) datasets;

Then run the following script for preprocessing:

```
python merge_datasets.py
```

## 🚀 Getting Started

1. Clone the repository to your local machine
2. Navigate to the project directory
3. You can use the provided script to run the code:
```
python run_recbole_cdr.py --config_files ./yaml/Douban.yaml
```
4. It is also available to run directly on Code Ocean at https://codeocean.com/capsule/7063643/tree/v1

## ❤️ Acknowledgement

Our code is developed based on [`RecBole-CDR`](https://github.com/RUCAIBox/RecBole-CDR), and we implement graph augmentation and graph pre-training based on [`PyGCL`](https://github.com/PyGCL/PyGCL) library.
