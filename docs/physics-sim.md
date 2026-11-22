# Physics simulation

We use [CARL](https://github.com/automl/CARL)'s BipedalWalker: a two-legged robot in a 2D world. It uses [Box2D](https://github.com/erincatto/box2d) to handle gravity, movement, and collisions.

[simulation.py](../simulation.py) sets up and displays the simulation.

We set CARL's `GRAVITY_Y` parameter to **-9.81 m/s²** for Earth and **-1.62 m/s²** for the Moon. The minus sign means gravity pulls down. Both runs use the same terrain seed (`42`) and other settings, so gravity is the main difference.

The robot is shown from the side, so its two legs can overlap. The thin red line is a distance sensor that scans the ground. We changed the Moon view to use gray ground and a dark sky; these colors don't affect the physics.

