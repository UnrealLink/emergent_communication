# Studying the joint role of partial observability and channel reliability in emergent communication

## Setup
```
conda env create -n emergent_communication -f environment.yml
python setup.py develop
conda activate causal
```

## Quick start

To try training agents on the finder env using DQN:

`python train\train_dqn.py --env finder --save test_dqn`

## Reproduce paper results

Each experiment done in the paper has its own separate branch:

    - solo_agent: 
    - easy_speaker:
    - easy_speaker_bias:
    - speaker_bias:

## Credits

Code for environments is taken from https://github.com/eugenevinitsky/sequential_social_dilemma_games