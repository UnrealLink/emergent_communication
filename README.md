## Experiment example

```
python train/train_a3c.py --env target --save target_hard_bias_view_2_0 --view-size 2 --seed 0 --horizon 1000000 --cpu-only
python metrics/measure_ic.py --env target --save target_hard_bias_view_2_0 --view-size 2 --seed 0 --cpu-only
```