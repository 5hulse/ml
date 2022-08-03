# pytorch_network.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 03 Aug 2022 13:03:53 BST

import os
from pathlib import Path
import time

import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
from import_data import TEST_DATA, TRAINING_DATA

mpl.use("tkAgg")
device = "cuda" if torch.cuda.is_available() else "cpu"


class Network(torch.nn.Module):

    def __init__(self, layer_dims) -> None:
        super().__init__()
        layers = []
        for l, r in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(torch.nn.Linear(l, r))
            layers.append(torch.nn.Sigmoid())
        layers.append(torch.nn.Softmax(dim=1))

        self.sigmoid_stack = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid_stack(x)


def train_loop(
    batch_size: int,
    features: torch.Tensor,
    labels: torch.Tensor,
    network: torch.nn.Module,
    loss_fn: torch.nn.MSELoss,
    optimiser: torch.optim.SGD,
) -> str:
    size = features.shape[0]
    batches = [
        (
            features[i : i + batch_size],
            labels[i : i + batch_size],
        )
        for i in range(0, size, batch_size)
    ]

    train_loss, acc = 0., 0.
    for batch_no, (X, y) in enumerate(batches):
        pred = network(X)
        loss = loss_fn(pred, y)
        train_loss += loss
        acc += accuracy(pred, y)

        # Backprop
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    train_loss /= len(batches)
    acc /= len(batches)

    return (
        f"Training loss: {train_loss:.5f}\n"
        f"Training accuracy: {(100 * acc):0.1f}%\n"
    )


def test_loop(
    features: torch.Tensor,
    labels: torch.Tensor,
    network: torch.nn.Module,
    loss_fn: torch.nn.MSELoss,
) -> str:
    with torch.no_grad():
        pred = network(features)
        test_loss = loss_fn(pred, labels).item()
        acc = accuracy(pred, labels)

    return (
        f"Test loss: {test_loss:.5f}\n"
        f"Test accuracy: {(100 * acc):0.1f}%\n"
    )


def accuracy(
    pred: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    return (
        pred.argmax(1) == torch.where(labels == 1)[1]
    ).type(torch.float).sum().item() / labels.shape[0]


def visualise_network_performance(
    network: Network,
    X: torch.Tensor,
    y: torch.Tensor,
) -> None:
    plt.ion()
    ticks = list(range(10))
    fig = plt.figure()
    ax0 = fig.add_axes([0.05, 0.35, 0.9, 0.6])
    ax1 = fig.add_axes([0.05, 0.1, 0.9, 0.2])
    ax1.set_xticks(ticks)
    ax1.set_ylim(0, 1)

    labels = torch.where(y == 1)[1]
    for image, label in zip(X, labels):
        label = label.item()
        pred = network(image[None, :])
        guess = pred.argmax(1).item()
        success = (guess == label)
        bar_colors = 10 * ["grey"]
        bar_colors[guess] = "green" if success else "red"
        imshow = ax0.imshow(image.reshape((28, 28)).cpu().detach().numpy())
        bar = ax1.bar(ticks, pred.cpu().detach().numpy().flatten(), color=bar_colors)

        if not success:
            txt = ax1.text(
                label, 0.5, str(label), ha="center", va="center", color="red",
                fontsize=18,
            )

        fig.canvas.flush_events()
        time.sleep(0.5)
        imshow.remove()
        bar.remove()

        if not success:
            txt.remove()


if __name__ == "__main__":
    TRAIN = True
    layer_dims = [784, 100, 10]

    layer_dim_str = "_".join([str(x) for x in layer_dims])
    path = Path(f"models/{layer_dim_str}.pth")

    if not path.parent.is_dir():
        os.mkdir(path.parent)

    X_test = torch.from_numpy(TEST_DATA.data.T).float().to(device)
    y_test = torch.from_numpy(TEST_DATA.labels.T).float().to(device)
    if TRAIN or not path.is_file():
        X_train = torch.from_numpy(TRAINING_DATA.data.T).float().to(device)
        y_train = torch.from_numpy(TRAINING_DATA.labels.T).float().to(device)
        lambda_ = 5.
        batch_size = 10
        epochs = 10
        network = Network(layer_dims).to(device)
        loss_fn = torch.nn.MSELoss()
        optimiser = torch.optim.SGD(network.parameters(), lr=lambda_)
        size = X_train.shape[0]

        for epoch_no in range(1, epochs + 1):
            msg = f"Epoch {epoch_no}\n"
            msg += (len(msg) - 1) * "-" + "\n"
            perm = torch.randperm(size)
            X_train = X_train[perm]
            y_train = y_train[perm]
            msg += train_loop(10, X_train, y_train, network, loss_fn, optimiser)
            msg += test_loop(X_test, y_test, network, loss_fn)
            print(msg)

        torch.save(network.state_dict(), path)

    else:
        network = Network(layer_dims).to(device)
        network.load_state_dict(torch.load(path))
        network.eval()

    visualise_network_performance(network, X_test, y_test)
