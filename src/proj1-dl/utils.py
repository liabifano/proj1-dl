import numpy as np
import torch
from torch.autograd import Variable


def compute_nb_errors(model, data_input, data_target, mini_batch_size):
    """
    Computer model's errors

    :param model: fitted neural network
    :param data_input: tensor with input
    :param data_target: tensor with input
    :param mini_batch_size: integer with batch size
    :return:
    """
    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        if b + mini_batch_size <= data_input.size(0):
            output = model(data_input.narrow(0, b, mini_batch_size))
            _, predicted_classes = torch.max(output.data, 1)
            for k in range(0, mini_batch_size):
                if data_target.data[b + k] != predicted_classes[k]:
                    nb_data_errors = nb_data_errors + 1
        else:
            output = model(data_input.narrow(0, b, data_input.size(0) - b))
            _, predicted_classes = torch.max(output.data, 1)
            for k in range(0, data_input.size(0) - b):
                if data_target.data[b + k] != predicted_classes[k]:
                    nb_data_errors = nb_data_errors + 1

    return nb_data_errors


def predict_classes(model, data_input, mini_batch_size):
    """
    Get predictions of model

    :param model: fitted neural network
    :param data_input: tensor with input
    :param mini_batch_size: integer with batch size
    :return: array of 0-1
    """
    preds = np.zeros((0,))

    for b in range(0, data_input.size(0), mini_batch_size):
        if b + mini_batch_size <= data_input.size(0):
            output = model(data_input.narrow(0, b, mini_batch_size))
            _, predicted_classes = torch.max(output.data, 1)
            predicted_classes = Variable(predicted_classes).data.numpy()

        else:
            output = model(data_input.narrow(0, b, data_input.size(0) - b))
            _, predicted_classes = torch.max(output.data, 1)
            predicted_classes = Variable(predicted_classes).data.numpy()

        preds = np.concatenate([preds, predicted_classes], axis=0)

    return preds.astype(int)


def predict_scores(model, data_input, mini_batch_size):
    """
    Get predictions of model and model's output

    :param model: fitted neural network
    :param data_input: tensor with input
    :param mini_batch_size: integer with batch size
    :return: array with 0-1 and neural network output
    """
    preds = np.zeros((0, 2))

    for b in range(0, data_input.size(0), mini_batch_size):
        if b + mini_batch_size <= data_input.size(0):
            output = model(data_input.narrow(0, b, mini_batch_size))
            predicted_scores, predicted_classes = torch.max(output.data, 1)
            predicted_scores = Variable(predicted_scores).data.numpy()
            predicted_classes = Variable(predicted_classes).data.numpy()
            predicted = np.concatenate([predicted_classes.reshape((len(predicted_classes), 1)),
                                        predicted_scores.reshape((len(predicted_scores), 1))], axis=1)


        else:
            output = model(data_input.narrow(0, b, data_input.size(0) - b))
            predicted_scores, predicted_classes = torch.max(output.data, 1)
            predicted_scores = Variable(predicted_scores).data.numpy()
            predicted_classes = Variable(predicted_classes).data.numpy()
            predicted = np.concatenate([predicted_classes.reshape((len(predicted_classes), 1)),
                                        predicted_scores.reshape((len(predicted_scores), 1))], axis=1)

        preds = np.concatenate([preds, predicted], axis=0)

    return preds
