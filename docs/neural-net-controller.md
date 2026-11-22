# Neural network controller

The controller is a small neural network. It takes 24 sensor readings, such as the robot's tilt, leg positions, and ground distances. A layer of 16 neurons processes those readings, and four outputs control the robot's motors.

[controller.py](../controller.py) defines the network and calculates its motor commands.

The outputs control the first leg's hip and knee, followed by the second leg's hip and knee. Each output tells its motor which direction to move and how much power to use.

The network's weights are numbers that decide how sensor readings affect movement. Neural networks are often trained with gradient descent, which adjusts weights using a measure of error. We use a genetic algorithm instead: it tests different weights and favors the ones that help the robot walk well. The same controller can be tested on Earth and the Moon.
