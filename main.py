from training import train
from plots import show_plots

LEARNING_RATE = 0.05
NESTEROV_GAMMA = 0.95

# First convolutional layer
FIRST_LAYER_SIZE = 5
FIRST_LAYER_FILTERS_COUNT = 4
# Second convolutional layer
SECOND_LAYER_SIZE = 5
SECOND_LAYER_FILTERS_COUNT = 6
# Simple layer
THIRD_LAYER_SIZE = 100
# Softmax layer
FOURTH_LAYER_SIZE = 10

PARAMETERS_FILE_NAME = "output/data"


def main() -> None:
    train(LEARNING_RATE,
          NESTEROV_GAMMA,
          PARAMETERS_FILE_NAME,
          FIRST_LAYER_SIZE,
          SECOND_LAYER_SIZE,
          FIRST_LAYER_FILTERS_COUNT,
          SECOND_LAYER_FILTERS_COUNT,
          THIRD_LAYER_SIZE,
          FOURTH_LAYER_SIZE)

    show_plots(PARAMETERS_FILE_NAME)


if __name__ == '__main__':
    main()
