# NFL Offensive Playcalling Optimization
A reinforcement learning approach to developing an optimal policy for NFL offensive playcalling

# Overview

The following project is based upon the idea of implementing various reinforcement learning algorithms in order to develop an optimal policy for calling plays. The policy development is based upon the [Kaggle Detailed NFL Play-by-Play](https://www.kaggle.com/maxhorowitz/nflplaybyplay2009to2016) dataset.

# Requirements

* Python >= 3.5 (type hinting)
* Pandas
* Kaggle API

# Setup

In order to setup this project, it's necessary to clone the repository and download the dataset from Kaggle

```bash
git clone https://github.com/jacobeturpin/nfl-offensive-playcalling-optimization.git
cd nfl-offensive-playcalling-optimization
mkdir data && kaggle datasets download maxhorowitz/nflplaybyplay2009to2016 -f "NFL Play by Play 2009-2018 (v5).csv" -p ./data/ --unzip
```