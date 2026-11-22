# Neural network controller

The genetic controller is a small neural network in [controller.py](../controller.py). It has 24 inputs, a layer of 16 hidden neurons, and four outputs. PPO uses its own neural network with the same inputs and outputs.

## What the robot senses

At each step, the [BipedalWalker simulation](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/bipedal_walker.py) sends these 24 readings to the controller:

- **Body (4):** body angle, turning speed, horizontal speed, and vertical speed.
- **Legs (10):** for each leg, the hip angle and speed, knee angle and speed, and whether the foot is touching the ground.
- **Ground sensors (10):** distance readings from rays that scan the ground beneath and ahead of the robot.

The simulator scales some speeds and represents each knee angle as its actual angle plus 1. Ground distances are fractions of the sensor's range. Gravity affects the robot's movement, but the gravity setting itself is not one of the 24 inputs.

## How it moves

The network turns those readings into four **motor commands**, in order: leg 1 hip, leg 1 knee, leg 2 hip, and leg 2 knee. Each command ranges from −1 to 1. Its sign chooses the motor's direction, and its size sets the maximum torque, or turning force, the motor can use. A command of 0 turns that motor off.

The network does not output target angles. The physics simulation applies the commands, moves the robot, and returns new joint angles and other readings. The controller uses those readings to choose the next commands.

## How it learns

The network's weights decide how sensor readings affect its commands. The genetic algorithm trains its controller by testing different weights and favoring those that help the robot walk well. PPO trains its own controller using rewards and gradient-based updates. Either controller can be tested on Earth and the Moon.
