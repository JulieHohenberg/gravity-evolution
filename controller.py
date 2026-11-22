"""A small neural network that turns sensor readings into motor commands: INFERENCE MODE"""

import numpy as np

INPUTS = 24
HIDDEN = 16
OUTPUTS = 4
WEIGHT_COUNT = INPUTS * HIDDEN + HIDDEN + HIDDEN * OUTPUTS + OUTPUTS

# take robot's sensor readings and the weights of the neural network to calculate the motor commands for the robot
def motor_commands(observation, weights):
    # The flat list lets the genetic algorithm change all weights together.
    split = INPUTS * HIDDEN
    hidden_weights = weights[:split].reshape(INPUTS, HIDDEN)
    hidden_bias = weights[split:split + HIDDEN]
    split += HIDDEN
    output_weights = weights[split:split + HIDDEN * OUTPUTS].reshape(HIDDEN, OUTPUTS)
    output_bias = weights[split + HIDDEN * OUTPUTS:]
    hidden = np.tanh(observation["obs"] @ hidden_weights + hidden_bias)
    return np.tanh(hidden @ output_weights + output_bias).astype(np.float32)
