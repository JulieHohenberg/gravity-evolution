# Genetic algorithm

The genetic algorithm trains the neural network by trying different weights and keeping the controllers that score best. [train_genetic.py](../train_genetic.py) runs the algorithm, using [controller.py](../controller.py) for motor commands (putting neural net in inference) and [simulation.py](../simulation.py) to create the environment.

## How it works

1. **Start:** Create 30 controllers. Each has 468 randomly chosen weights and biases, which control how its neural network moves the robot.
2. **Test:** Run each controller on the same course. At each step, the network uses the robot's sensor readings to choose four motor commands. The simulation returns new readings and a reward. A run ends when the robot falls, finishes, or reaches a time limit, with a maximum of 1,600 steps.
3. **Score:** Add the rewards from every step: **fitness = reward₁ + reward₂ + ... + rewardₙ**. A higher score is better.
4. **Select:** Choose the six controllers with the highest scores as parents. Keep track of the best controller seen across all generations.
5. **Create the next generation:** Carry the best controller forward unchanged. For each of the other 29, randomly pick two parents and choose each weight from one parent or the other. Each weight then has a 10% chance of a small random change, called a mutation.
6. **Repeat:** Test the new generation and repeat for 30 generations by default. Save the best controller's weights to a file.

The [BipedalWalker simulation](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/bipedal_walker.py) rewards forward movement and penalizes tilting and motor use. The motor penalty encourages the robot to use less energy. Falling gives a reward of -100 for that step and ends the run.

Training uses the same terrain seed (`42`) for each controller. Run `python train_genetic.py` for Earth or add `--planet Moon` for the Moon. Use `--generations 10` to change the number of generations and `--overwrite` to replace saved weights.

Each training run gets its own folder in `runs/`. It contains the best controller's weights from each generation (`generation-001.npy`, etc.), a `scores.csv` with each generation's best, average, worst, and overall best scores, and a `population-scores.csv` with every controller's score. `settings.json` records the training settings. To watch a generation, run `python simulation.py --controller` followed by its `.npy` file path.
