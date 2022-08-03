# pytorch_network.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 03 Aug 2022 01:47:50 BST

import torch
from import_data import TEST_DATA, TRAINING_DATA


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


if __name__ == "__main__":
    X_train = torch.from_numpy(TRAINING_DATA.data.T).float().to(device)
    y_train = torch.from_numpy(TRAINING_DATA.labels.T).float().to(device)
    X_test = torch.from_numpy(TEST_DATA.data.T).float().to(device)
    y_test = torch.from_numpy(TEST_DATA.labels.T).float().to(device)
    lambda_ = 5.
    batch_size = 10
    epochs = 30
    network = Network([784, 100, 10]).to(device)
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
