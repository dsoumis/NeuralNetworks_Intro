import sys

import numpy
import pandas
from keras import optimizers, losses, metrics
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error


def assign_values_from_arguments():
    inp = ""
    mod = ""
    act = ""

    if len(sys.argv) != 7:
        print("Please re-run with correct arguments.")
        sys.exit()

    for i in range(len(sys.argv)):
        if sys.argv[i] == "-i":
            inp = sys.argv[i+1]
        elif sys.argv[i] == "-m":
            mod = sys.argv[i + 1]
        elif sys.argv[i] == "-a":
            act = sys.argv[i + 1]
    return inp, mod, act


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = numpy.array(y_true), numpy.array(y_pred)
    for index, row in enumerate(y_true):
        y_true[index] = numpy.mean(row)
    mape = numpy.abs((y_true - y_pred) / y_true)
    return numpy.mean(mape[numpy.isfinite(mape)]) * 100
    # The following line is to use as mean the whole file and not row by row
    # return numpy.mean(numpy.abs((numpy.mean(y_true) - y_pred) / numpy.mean(y_true))) * 100


if __name__ == '__main__':
    """
    The main function called when predict.py is run from the command line
    
    >Execute: python3.7 predict.py -i path_to:nn_representations.csv -m path_to:WindDenseNN.h5 -a path_to:actual.csv
    """
    inputFile, inputModel, actualFile = assign_values_from_arguments()
    model = load_model(inputModel)
    # We don't care for which optimizer to use as we use a pre-trained model
    model.compile(optimizer=optimizers.Adam(0.01), loss=losses.mape,
                  metrics=[metrics.mae, metrics.mape])

    data = pandas.read_csv(inputFile, header=None)
    timestamps = data.loc[:, 0]
    data = data.drop(data.columns[0], axis=1)

    labels = pandas.read_csv(actualFile, header=None)
    labels = labels.drop(labels.columns[0], axis=1)

    model.evaluate(data, labels, batch_size=32)

    result = model.predict(data, batch_size=32)

    beginning_of_file = 'MAE: ' + str(mean_absolute_error(labels, result)) + ' MAPE: '\
                        + str(mean_absolute_percentage_error(labels, result)) +\
                        ' MSE: ' + str(mean_squared_error(labels, result))
    result = numpy.c_[timestamps.to_numpy(), result]
    numpy.savetxt("predicted.csv", result, delimiter="\t", fmt='%s', header=beginning_of_file,
                  comments='')


