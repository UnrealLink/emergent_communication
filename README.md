## Experiment example

```
python train/train_a3c.py --env target --save target_view_2_noise_3_0 --view-size 2 --seed 0 --horizon 1000000 --cpu-only --noise 0.3
python measure/measure.py --env target --save target_view_2_noise_3_0 --view-size 2 --seed 0 --cpu-only --noise 0.3
```