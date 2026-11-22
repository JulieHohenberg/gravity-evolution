# Physics simulation

We use [CARL](https://github.com/automl/CARL)'s BipedalWalker: a two-legged robot in a 2D world. It uses [Box2D](https://github.com/erincatto/box2d) to handle gravity, movement, and collisions.

[simulation.py](../simulation.py) sets up and displays the simulation.

We set CARL's `GRAVITY_Y` parameter to **-9.81 m/s²** for Earth and **-1.62 m/s²** for the Moon. The minus sign means gravity pulls down. Both runs use the same terrain seed (`42`) and other settings, so gravity is the main difference.

The robot is shown from the side, so its two legs can overlap. The thin red line is a distance sensor that scans the ground. We changed the Moon view to use gray ground and a dark sky; these colors don't affect the physics.

## Rewards

The simulator rewards forward movement and penalizes tilting, motor use, and falling. We noticed some trained robots stopped moving and reached the time limit before the finish. To encourage movement and finishing, we changed two rewards for both training methods:

- **Motor use:** We keep 25% of the original motor penalty, so moving the legs costs fewer points.
- **Finishing:** Crossing the finish line adds `100 + 0.1 × (1,600 − steps taken)` points. Finishing sooner earns more points. A robot that falls or runs out of time gets no finish bonus.

For example, finishing in 1,000 steps adds 160 points; finishing in 1,500 adds 110. The changes encourage progress, though training is needed to see how the robots respond.

Each training run's `settings.json` records its reward version: `original` uses BipedalWalker's default rewards, and `finish-and-speed-v1` uses the changes above. Compare results with that difference in mind.

