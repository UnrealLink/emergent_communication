# Studying the joint role of partial observability and channel reliability in emergent communication

## Setup
```
conda env create -n emergent_communication -f environment.yml
conda activate emergent_communication
python setup.py develop
```

If you are on Windows, use the env_windows.yml file instead.

## Quick start

To try training agents on the finder environment using DQN:

`python train\train_dqn.py --env finder --save test_dqn`

## Reproduce paper results

Each experiment done in the paper has its own separate branch:

    - solo_agent: only the listener is trained, receiving perfect messages from a hard-coded speaker
    - easy_speaker: a simple mlp speaker is added, taking the perfect message as input
    - easy_speaker_bias: a positive signalling bias is used to help train the speaker
    - speaker_bias: the speaker now receives only the visual input

## Credits

Code for environments is taken from https://github.com/eugenevinitsky/sequential_social_dilemma_games
