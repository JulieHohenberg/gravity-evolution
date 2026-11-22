# Reinforcement learning

We use Proximal Policy Optimization (PPO) from [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) to train a walking controller. [train_rl.py](../train_rl.py) sets up training, and [simulation.py](../simulation.py) displays the trained model.

Stable-Baselines3 provides PPO and its neural network controller. Its built-in `model.learn()` runs the training loop and updates the network. Our code connects it to the CARL simulation, selects gravity, and saves models and scores. This network is separate from the one in [controller.py](../controller.py), which we use for the genetic algorithm.

## How PPO works

PPO teaches a controller through practice and feedback. The controller (a POLICY in this case) is a neural network that decides what to do from what it senses. For our robot, it takes 24 sensor readings, including tilt, leg positions, and ground distances, and chooses four motor commands as output.

1. **Practice:** The robot tries motor commands in the simulation. During training, its choices include some randomness so it can try different movements.
2. **Get feedback:** Each step gives the robot a reward. Moving forward earns points; tilting, using motor power, and falling can lower the score. The robot also receives new sensor readings for its next decision.
3. **Judge the actions:** PPO estimates which actions led to better or worse results than expected. It considers later rewards too (THINKING INTO FUTURE -> think like a good chess player), so an action can get credit for helping the robot move forward afterward.
4. **Update the controller:** PPO uses gradient-based optimization to change the network's weights, making promising actions more likely and less helpful ones less likely. It limits the incentive for large changes so the controller doesn't change too suddenly.
5. **Repeat:** The robot practices with the updated controller and PPO learns from the new results.

Unlike our genetic algorithm, which tests a population of different controllers and breeds the better ones, PPO improves a controller using feedback from its own practice. Think of this like one intelligent robot being born into the world and growing up and learning, rather than a controller being selected through the natural selection of many generations of naive robots. Both use the same sensor readings, motor commands, gravity settings, and simulation rewards, but their neural networks have different structures.

After some robots stalled before the finish, we reduced the motor penalty and added a bonus for finishing quickly. Both PPO and the genetic algorithm use the same changes; see [physics simulation rewards](physics-sim.md#rewards).

## Rewards and settings

PPO learns from the rewards returned at each simulation step. The [BipedalWalker code](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/bipedal_walker.py) supplies the original rewards; our changes are in `WalkingRewards` in [simulation.py](../simulation.py).

### BipedalWalker defaults

The simulator first calculates a progress value using the robot's horizontal position and body tilt:

```text
progress = 130 × position / 30 − 5 × |tilt|
step reward = change in progress − motor cost
```

Here, `30` is the simulator's default scale and `|tilt|` is the size of the body angle, regardless of direction. Moving forward increases progress, while leaning further decreases it. The step reward uses the **change** in progress since the previous step.

The default motor cost is `0.028 × (|motor 1| + |motor 2| + |motor 3| + |motor 4|)`, with each motor's magnitude capped at 1. On a fall, the simulator sets that step's reward to **−100** and ends the run. Crossing the finish line ends the run but has no separate default bonus.

### Our changes

Some robots stopped moving and reached the time limit. We made these changes to encourage them to keep moving and reach the end:

- **Lower motor cost:** We keep 25% of the default penalty. With the current motor settings, the new cost is `0.007 × (|motor 1| + |motor 2| + |motor 3| + |motor 4|)`, with each magnitude capped at 1.
- **Finish and speed bonus:** If the robot crosses the finish line without falling, we add `100 + 0.1 × max(0, 1,600 − steps taken)` to that step's reward. Finishing in 1,000 steps adds **160** points; finishing in 1,500 adds **110**. Falling or timing out earns no finish bonus.

The falling penalty stays at **−100**. These reward changes also apply to the genetic algorithm, so both methods train with the same rules.

### PPO training settings

We choose the planet and use seed `42`. By default, we train for **1,000,000 steps**, collect **2,048 steps** between PPO updates, and save and evaluate every **20,480 steps**. Stable-Baselines3 supplies PPO and its other settings, including a **0.0003 learning rate** and **0.2 clip range**. The learning rate controls the size of weight adjustments; clipping helps discourage overly large policy changes.

## Train and watch

Run `python train_rl.py --planet Earth` or `python train_rl.py --planet Moon`. The default is one million simulation steps; use `--steps 20480` for a shorter run. PPO collects 2,048 steps at a time, so it rounds up to finish the current batch.

Each run has its own folder in `runs/`. It contains the final model, settings, training episode rewards, and checkpoints and evaluation scores every 20,480 steps. Each evaluation tests the controller once and saves the best evaluated model. Load a saved model with `python simulation.py --ppo` followed by its `.zip` file path. Training and evaluation use terrain seed `42`.
