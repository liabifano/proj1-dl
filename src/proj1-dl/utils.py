import torch

def compute_nb_errors(model, data_input, data_target, mini_batch_size):
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