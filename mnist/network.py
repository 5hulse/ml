# network.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 03 Aug 2022 12:57:27 BST

import abc
from collections import deque
from typing import Iterable, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from import_data import Dataset, TRAINING_DATA, TEST_DATA

mpl.use("tkAgg")


class Activation(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def fn(*args):
        pass

    @abc.abstractmethod
    def derivative(*args):
        pass


class Sigmoid(Activation):

    @staticmethod
    def fn(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    @classmethod
    def derivative(cls, z: np.ndarray) -> np.ndarray:
        sigmoid = cls.fn(z)
        return sigmoid * (1 - sigmoid)


class Cost(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def fn(*args):
        pass

    @abc.abstractmethod
    def delta(*args):
        pass


class RSS(Cost):

    @staticmethod
    def fn(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 0.5 * (np.sum(y - y_hat, axis=0) ** 2)

    @staticmethod
    def delta(z: np.ndarray, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (y_hat - y) * Sigmoid.derivative(z)


class CrossEntropy(Cost):

    @staticmethod
    def fn(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sum(
            -y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat),
            axis=0,
        )

    @staticmethod
    def delta(z: np.ndarray, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        return y_hat - y


class Network:
    def __init__(
        self,
        layers: Iterable[int],
        activation_functions: Optional[Iterable[Activation]] = None,
    ) -> None:
        self.layers = layers
        self.nlayers = len(self.layers)

        if activation_functions is None:
            activation_functions = (self.nlayers - 1) * [Sigmoid]
        self.activation_functions = activation_functions

        self.weights = [np.random.normal(loc=0.0, scale=1.0 / np.sqrt(l), size=(r, l))
                        for l, r in zip(self.layers[:-1], self.layers[1:])]
        self.biases = [np.random.normal(loc=0.0, scale=1.0, size=(s, 1))
                       for s in self.layers[1:]]

    def __len__(self) -> int:
        return self.nlayers

    def feed_forward(self, dataset: Dataset) -> np.ndarray:
        activations = [dataset.data]
        for w, b, f in zip(self.weights, self.biases, self.activation_functions):
            activations.append(
                f.fn(w @ activations[-1] + np.repeat(b, dataset.size, axis=1))
            )
        return activations[-1]

    def back_propagation(
        self,
        batch: Dataset,
        cost: Cost,
    ) -> Tuple[deque[np.ndarray], deque[np.ndarray]]:
        pre_activations = []
        activations = [batch.data]

        for w, b, f in zip(self.weights, self.biases, self.activation_functions):
            pre_activations.append(
                w @ activations[-1] + np.repeat(b, batch.size, axis=1)
            )
            activations.append(f.fn(pre_activations[-1]))

        nabla_b = deque()
        nabla_w = deque()
        g = cost.delta(pre_activations[-1], activations[-1], batch.labels)
        for i, (w, z, a, f) in enumerate(
            zip(
                reversed(self.weights),
                reversed(pre_activations),
                reversed(activations[:-1]),
                reversed(self.activation_functions),
            )
        ):
            if i != 0:
                g *= f.derivative(z)
            nabla_b.appendleft(np.mean(g, axis=1, keepdims=True))
            nabla_w.appendleft(np.mean(np.einsum("ik,jk->ijk", g, a), axis=2))
            g = w.T @ g

        return nabla_b, nabla_w

    def stochastic_gradient_descent(
        self,
        training_data: Dataset,
        epochs: int,
        batch_size: int,
        eta: float,
        test_data: Optional[Dataset] = None,
        cost: Cost = RSS,
        regularisation: str = "L2",
        lambda_: float = 0.0,
        visualise: bool = True,
    ) -> None:
        if visualise:
            plt.ion()
            ticks = list(range(10))
            fig = plt.figure()
            ax0 = fig.add_axes([0.05, 0.35, 0.9, 0.6])
            ax1 = fig.add_axes([0.05, 0.1, 0.9, 0.2])
            ax1.set_xticks(ticks)
            ax1.set_ylim(0, 1)

        training_costs = []
        training_accuracies = []
        if test_data is not None:
            test_costs = []
            test_accuracies = []

        for epoch in range(1, epochs + 1):
            batches = training_data.make_batches(batch_size)
            training_cost = 0.0
            training_successes = 0

            for batch_no, batch in enumerate(batches):
                if visualise and batch_no % 10 == 0:
                    sample = batch.get_sample(0)
                    title = fig.text(
                        0.05, 0.75, f"Epoch {epoch}\nBatch {batch_no}"
                    )
                    correct_value = np.where(sample.labels == 1.)[0][0]
                    distribution = self.feed_forward(sample)
                    distribution /= np.sum(distribution)
                    color = 10 * ["grey"]
                    max_ = np.argmax(distribution)
                    success = (sample.labels[max_] == 1.)[0]
                    color[max_] = "green" if success else "red"
                    imshow = ax0.imshow(sample.data.reshape((28, 28)))
                    bar = ax1.bar(ticks, distribution.flatten(), color=color)

                    if not success:
                        txt = ax1.text(
                            correct_value, 0.5, correct_value, ha="center",
                            va="center", color="red", fontsize=20,
                        )

                    fig.canvas.flush_events()
                    import time
                    time.sleep(0.05)
                    imshow.remove()
                    bar.remove()
                    title.remove()
                    if not success:
                        txt.remove()

                tc = batch.size * self.cost_average(batch, cost) / training_data.size
                if regularisation == "L2":
                    tc += (
                        (0.5 * batch.size * lambda_ / training_data.size ** 2) *
                        (np.sum([np.sum(w ** 2) for w in self.weights]))
                    )
                elif regularisation == "L1":
                    tc += (
                        (0.5 * batch.size * lambda_ / training_data.size ** 2) *
                        (np.sum([np.sum(np.abs(w)) for w in self.weights]))
                    )
                training_cost += tc

                training_successes += self.evaluate(batch)
                nabla_b, nabla_w = self.back_propagation(batch, cost)
                for i, (nb, nw) in enumerate(zip(nabla_b, nabla_w)):
                    self.biases[i] -= eta * nb
                    self.weights[i] -= eta * nw
                    if regularisation == "L2":
                        self.weights[i] -= (
                            eta * (lambda_ / training_data.size) * self.weights[i]
                        )
                    elif regularisation == "L1":
                        self.weights[i] -= (
                            eta * (lambda_ / training_data.size) *
                            np.sign(self.weights[i])
                        )

            training_costs.append(training_cost)
            training_accuracies.append((training_successes / training_data.size) * 100)

            msg = f"Epoch {epoch}"
            msg += f"\n{'-' * len(msg)}\n"
            msg += (
                f"Training cost average: {training_costs[-1]:.5g}\n"
                f"Training accuracy: {training_accuracies[-1]:.5g}%\n"
            )
            if test_data is not None:
                tc = self.cost_average(test_data, cost)

                if regularisation == "L2":
                    tc += (
                        (lambda_ / (2 * test_data.size)) *
                        (np.sum([np.sum(w ** 2) for w in self.weights]))
                    )
                elif regularisation == "L1":
                    tc += (
                        (lambda_ / (2 * test_data.size)) *
                        (np.sum([np.sum(np.abs(w)) for w in self.weights]))
                    )

                test_costs.append(tc)
                test_successes = self.evaluate(test_data)
                test_accuracies.append((test_successes / test_data.size) * 100)
                msg += (
                    f"Test cost average: {test_costs[-1]:.5g}\n"
                    f"Test accuracy: {test_accuracies[-1]:.5g}%\n"
                )
            msg += "\n"
            print(msg)

    def cost_average(
        self,
        dataset: Dataset,
        cost: Cost,
    ) -> float:
        return np.sum(
            cost.fn(self.feed_forward(dataset), dataset.labels)
        ) / dataset.size

    def evaluate(
        self,
        dataset: Dataset,
    ) -> int:
        return (
            np.argmax(dataset.labels, axis=0) ==
            np.argmax(self.feed_forward(dataset), axis=0)
        ).sum()


if __name__ == "__main__":
    net = Network([784, 100, 10])
    net.stochastic_gradient_descent(
        TRAINING_DATA, 30, 20, 0.5, test_data=TEST_DATA, cost=RSS,
        regularisation=None, lambda_=5.0,
    )
