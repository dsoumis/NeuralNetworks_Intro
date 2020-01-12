import sys

import keras
import numpy
import pandas
from keras import layers
from keras.models import load_model


def assign_values_from_arguments():
    inp = ""
    mod = ""

    if len(sys.argv) != 5:
        print("Please re-run with correct arguments.")
        sys.exit()

    for i in range(len(sys.argv)):
        if sys.argv[i] == "-i":
            inp = sys.argv[i+1]
        elif sys.argv[i] == "-m":
            mod = sys.argv[i + 1]
    return inp, mod


if __name__ == '__main__':
    """
    The main function called when predict.py is run from the command line

    >Execute: python3.7 new_representation.py -i path_to:nn_representations.csv -m path_to:WindDenseNN.h5
    """
    inputFile, inputModel = assign_values_from_arguments()
    weights = load_model(inputModel).layers[0].get_weights()
    # If we uncomment the following line, we notice that relu is used as activation to this layer
    # print(load_model(inputModel).layers[0].output)
    model = keras.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(128,)))
    model.set_weights(weights)

    data = pandas.read_csv(inputFile, header=None)
    timestamps = data.loc[:, 0]
    data = data.drop(data.columns[0], axis=1)

    # model.summary()

    result = numpy.c_[timestamps.to_numpy(), model.predict(data, batch_size=32)]
    beginning_of_file = 'vectors'
    numpy.savetxt("new_representation.csv", result, delimiter="\t", fmt='%s', header=beginning_of_file, comments='')
