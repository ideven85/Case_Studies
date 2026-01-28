"""
Utils for running our PyTorch experiments
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Read the simple 2D dataset files
def get_data_loader(name, batch_size, validation_split=None):
    try:
        loaded_data = np.loadtxt(name, skiprows=0, delimiter=" ")
    except:
        assert (
            validation_split is None
        ), "Please make sure there is a folder `./data` on this file's path"
        return None, None
    # print(loaded_data)

    np.random.shuffle(loaded_data)  # shuffle the data
    # The data uses ROW vectors for a data point, that's what PyTorch assumes.
    _, d = loaded_data.shape
    X = loaded_data[:, 0 : d - 1]
    Y = loaded_data[:, d - 1 : d]
    y = Y.T[0]

    class SimpleCustomBatch:
        def __init__(self, data):
            transposed_data = list(zip(*data))
            self.X = torch.stack(transposed_data[0], 0)
            self.y = torch.stack(transposed_data[1], 0)

    classes = set(y)
    if classes == set([-1.0, 1.0]):
        print("Convert from -1,1 to 0,1")
        y = 0.5 * (y + 1)

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    classes = set(y.numpy())

    if validation_split is not None:
        print("Using validation split")
        N = X.shape[0]
        val_size = int(validation_split * N)
        index = list(range(N))
        np.random.shuffle(index)

        print("train_size", N - val_size, "val_size", val_size)

        train_index = index[: N - val_size]
        inps, tgts = X[train_index], y[train_index]
        dataset = TensorDataset(inps, tgts)
        train_loader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=SimpleCustomBatch
        )
        print("Loading train X", inps.shape, "y", tgts.shape, "classes", classes)

        val_index = index[-val_size:]
        inps, tgts = X[val_index], y[val_index]
        dataset = TensorDataset(inps, tgts)
        val_loader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=SimpleCustomBatch
        )
        print("Loading val X", inps.shape, "y", tgts.shape, "classes", classes)

        return train_loader, val_loader, len(classes)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=SimpleCustomBatch)

    print("Loading X", X.shape, "y", y.shape, "classes", classes)
    return loader, len(classes)


def dataset_paths(data_name):
    return [
        "./data/data" + data_name + "_" + suffix + ".csv"
        for suffix in ("train", "validate", "test")
    ]


def call_model(mode, model, data_iter, optimizer, criterion):
    epoch_loss = []
    hits = []
    items = []

    if mode == "train":
        model.train()
        grad_mode = torch.enable_grad()
    else:
        model.eval()
        grad_mode = torch.no_grad()

    with grad_mode:

        for batch in data_iter:
            X, y = batch.X, batch.y

            if mode == "train":
                # zero the parameter gradients
                optimizer.zero_grad()

            # forward
            y_hat = model(X)
            batch_loss = criterion(y_hat, y.long())

            if mode == "train":
                # backward + optimize
                batch_loss.backward()
                optimizer.step()

            epoch_loss.append(batch_loss.item())
            hits.append((y_hat.argmax(1) == y.long()).sum())
            items.append(X.shape[0])

        loss = np.sum(epoch_loss) / np.sum(items)
        acc_score = np.sum(hits) / np.sum(items)
        return loss, acc_score


def model_fit(
    model,
    train_iter,
    epochs,
    optimizer,
    criterion,
    validation_split,
    validation_iter,
    history,
    verbose,
):

    av_train_loss, av_train_acc, av_vali_loss, av_vali_acc = [], [], [], []
    for epoch in range(epochs):
        train_loss, train_acc_score = call_model(
            "train", model, train_iter, optimizer, criterion
        )
        vali_loss, vali_acc_score = call_model(
            "vali", model, validation_iter, optimizer, criterion
        )
        if verbose:
            print(
                "epoch: {} | TRAIN: loss {} acc {} | VALI: loss {} acc {}".format(
                    epoch,
                    round(train_loss, 5),
                    round(train_acc_score, 5),
                    round(vali_loss, 5),
                    round(vali_acc_score, 5),
                )
            )
        if history is not None:
            history["epoch_loss"].append(train_loss)
            history["epoch_val_loss"].append(vali_loss)
            history["epoch_acc"].append(train_acc_score)
            history["epoch_val_acc"].append(vali_acc_score)
        av_train_loss.append(train_loss)
        av_train_acc.append(train_acc_score)
        av_vali_loss.append(vali_loss)
        av_vali_acc.append(vali_acc_score)

    return (
        (np.mean(av_train_loss), np.mean(av_train_acc)),
        (np.mean(av_vali_loss), np.mean(av_vali_acc)),
    )


def model_evaluate(model, test_iter, criterion):
    vali_loss, vali_acc_score = call_model("vali", model, test_iter, None, criterion)
    return vali_loss, vali_acc_score


def run_pytorch(
    train_iter, val_iter, test_iter, layers, epochs, split=0, verbose=True, history=None
):
    # Model specification
    for layer in layers:
        if type(layer) == nn.Linear:
            layer.reset_parameters()
    model = nn.Sequential(*layers)

    # Define the optimization
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # Fit the model
    train_m, vali_m = model_fit(
        model,
        train_iter,
        epochs=epochs,
        optimizer=optimizer,
        criterion=criterion,
        validation_split=split,
        validation_iter=val_iter,
        history=history,
        verbose=verbose,
    )
    if verbose:
        print()

    train_loss, train_acc = train_m
    vali_loss, val_acc = vali_m

    # Evaluate the model on test data, if any
    if test_iter is not None:
        test_loss, test_acc = model_evaluate(model, test_iter, criterion)
        print(
            "\nLoss on test set:"
            + str(test_loss)
            + " Accuracy on test set: "
            + str(test_acc)
        )
    else:
        test_acc = None
    return model, val_acc, test_acc
