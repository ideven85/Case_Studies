import numpy as np

from sequential_expected import (
    test_1_values,
    test_1_sgd_values,
    test_2_values,
    test_2_sgd_values,
)
from test_suite import run_test_suite
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from lib import rv
import optim as our_optim
from plotting import tidy_plot, plot_points, plot_fun, plot_heat
from utils import get_data_loader, dataset_paths, run_pytorch

### Optimizers


def f(x):
    return (x - 2.0) * (x - 3.0) * (x + 3.0) * (x + 1.0)


def fdf(x):
    return f(x), 9.0 - (22.0 * x) - (3.0 * x**2) + (4.0 * x**3)


def gd_with_optim(
    f_df,
    x0,
    step_size=0.01,
    step_size_fn=None,
    max_iter=1000,
    optim_cls=our_optim.GD,
    eps=1e-5,
    hook=None,
):
    """
    Runs gradient descent with our custom optimizer class to compute the gradient.
    See below for function for your experiments
    """
    max_iter = min(max_iter, 1000)
    prev_x = x0
    fs = []
    xs = []

    optim = optim_cls(shape=x0.shape)

    for i in range(max_iter):
        prev_f, prev_grad = f_df(prev_x)
        if prev_grad is None:
            prev_grad = num_grad(lambda x: f_df(x)[0])(prev_x)

        fs.append(float(prev_f))
        xs.append(prev_x)

        step = step_size_fn(i) if step_size_fn else step_size

        assert prev_x.shape == prev_grad.shape

        x = prev_x - step * optim.get_grad_step(i + 1, prev_grad)

        if hook:
            hook(x)
        if np.all(abs(x - prev_x) < eps):
            f, _ = f_df(x)
            fs.append(float(f))
            xs.append(x)
            return x, fs, xs
        prev_x = x
    return x, fs, xs


def run_gd_with_optim(
    step_size=0.01,
    init_val=1.0,
    num_steps=10,
    step_size_fn=None,
    optim_cls=our_optim.Adam,
):
    """
    Use this function to run experiments with different optimizers (GD and Adam)
    """
    init_weights = rv([init_val])
    w, js, ws = gd_with_optim(
        fdf,
        init_weights,
        optim_cls=optim_cls,
        step_size=step_size,
        step_size_fn=step_size_fn,
        max_iter=num_steps,
    )
    text_output = "objective", js[-1], "thetas", w
    nax = tidy_plot(
        -4,
        4,
        -25,
        25,
        xlabel="\u03f4",
        ylabel="f(\u03f4)",
        title="step size = " + str(step_size),
        center=True,
    )
    plot_points(
        [float(w) for w in ws],
        [float(j) for j in js],
        nax,
        mark_initial=True,
        mark_final=True,
    )
    plot_fun(nax, lambda w: fdf(w)[0], -4, 4)
    plt.show()
    return text_output


### PyTorch


def run_pytorch_2d(
    data_name,
    layers,
    epochs,
    split=0.25,
    display=True,
    verbose=True,
    trials=1,
    batch_size=32,
    return_history=False,
):
    """
    Trains and evaluates a given model `layers` with the dataset specified by `data_name`
    You should use the `archs` function above to create your model.

    Example usage:

        layers = [nn.Linear(in_features=2, out_features=2, bias=True), nn.ReLU(), nn.Linear(in_features=2, out_features=2, bias=True)]

        # this will train on dataset "1"
        model = run_pytorch_2d("1", layers, epochs=200, trials = 1, verbose=True, display=True)

    #If you don't care about the return values, throw them into a trash variable:

        _ = run_pytorch_2d("2", layers, epochs=200, display=False, verbose=False, trials=5)
    """
    print("Pytorch FC: dataset=", data_name)
    train_dataset_path, val_dataset_path, test_dataset_path = dataset_paths(data_name)
    # Load the datasets
    train_iter, num_classes = get_data_loader(train_dataset_path, batch_size)
    val_iter, num_classes = get_data_loader(val_dataset_path, batch_size)
    test_iter, num_classes = get_data_loader(test_dataset_path, batch_size)

    if val_iter is None:
        # Use split
        print("Use split", train_iter)
        assert split > 0, "`split` must be > 0"
        train_iter, val_iter, num_classes = get_data_loader(
            train_dataset_path, batch_size, split
        )

    val_acc, test_acc = 0, 0
    X_train = torch.cat([batch.X for batch in train_iter], 0)
    y_train = torch.cat([batch.y for batch in train_iter], 0)

    for trial in range(trials):
        trial_history = {
            "epoch_loss": [],
            "epoch_val_loss": [],
            "epoch_acc": [],
            "epoch_val_acc": [],
        }

        if verbose:
            print("\n")
        print(f"# Trial {trial}")

        # Run the model
        (
            model,
            vacc,
            tacc,
        ) = run_pytorch(
            train_iter,
            val_iter,
            test_iter,
            layers,
            epochs,
            split=split,
            verbose=verbose,
            history=trial_history,
        )

        val_acc += vacc if vacc else 0
        test_acc += tacc if tacc else 0
        if display:
            # plot classifier landscape on training data
            plot_heat(X_train, y_train, model)
            plt.title("Training data")
            plt.show()
            if test_iter is not None:
                # plot classifier landscape on testing data
                X_test = torch.cat([batch.X for batch in test_iter], 0)
                y_test = torch.cat([batch.y for batch in test_iter], 0)
                plot_heat(X_test, y_test, model)
                plt.title("Testing data")
                plt.show()
            # Plot epoch loss
            plt.figure(facecolor="white")
            plt.plot(
                range(epochs), trial_history["epoch_loss"], label="epoch_train_loss"
            )
            plt.plot(
                range(epochs), trial_history["epoch_val_loss"], label="epoch_val_loss"
            )
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title("Epoch val_loss and loss")
            plt.legend()
            plt.show()
            # Plot epoch accuracy
            plt.figure(facecolor="white")
            plt.plot(range(epochs), trial_history["epoch_acc"], label="epoch_train_acc")
            plt.plot(
                range(epochs), trial_history["epoch_val_acc"], label="epoch_val_acc"
            )
            plt.xlabel("epoch")
            plt.ylabel("accuracy")
            plt.legend()
            plt.title("Epoch val_acc and acc")
            plt.show()
    if val_acc:
        print("\nAvg. validation accuracy:" + str(val_acc / trials))
    if test_acc:
        print("\nAvg. test accuracy:" + str(test_acc / trials))
    if return_history:
        return model, trial_history
    return model


class Module:
    def sgd_step(self, lrate):
        pass  # For modules w/o weights


### Data sets
# Note that these are not the same as in hw5,
# as y is augmented to look like the outputs from softmax


def for_softmax(y):
    return np.vstack([1 - y, y])


def super_simple_separable_through_origin():
    X = np.array([[2, 3, 9, 12], [5, 1, 6, 5]])
    y = np.array([[1, 0, 1, 0]])
    return X, for_softmax(y)


def super_simple_separable():
    X = np.array([[2, 3, 9, 12], [5, 2, 6, 5]])
    y = np.array([[1, 0, 1, 0]])
    return X, for_softmax(y)


def xor():
    X = np.array([[1, 2, 1, 2], [1, 2, 2, 1]])
    y = np.array([[1, 1, 0, 0]])
    return X, for_softmax(y)


def xor_more():
    X = np.array([[1, 2, 1, 2, 2, 4, 1, 3], [1, 2, 2, 1, 3, 1, 3, 3]])
    y = np.array([[1, 1, 0, 0, 1, 1, 0, 0]])
    return X, for_softmax(y)


def hard():
    X = np.array(
        [
            [
                -0.23390341,
                1.18151883,
                -2.46493986,
                1.55322202,
                1.27621763,
                2.39710997,
                -1.3440304,
                -0.46903436,
                -0.64673502,
                -1.44029872,
                -1.37537243,
                1.05994811,
                -0.93311512,
                1.02735575,
                -0.84138778,
                -2.22585412,
                -0.42591102,
                1.03561105,
                0.91125595,
                -2.26550369,
            ],
            [
                -0.92254932,
                -1.1030963,
                -2.41956036,
                -1.15509002,
                -1.04805327,
                0.08717325,
                0.8184725,
                -0.75171045,
                0.60664705,
                0.80410947,
                -0.11600488,
                1.03747218,
                -0.67210575,
                0.99944446,
                -0.65559838,
                -0.40744784,
                -0.58367642,
                1.0597278,
                -0.95991874,
                -1.41720255,
            ],
        ]
    )
    y = np.array(
        [
            [
                1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
            ]
        ]
    )
    return X, for_softmax(y)


### Test cases


def test_dReLU_dz(fn):
    """
    Tests dReLU_dz(): dReLU_dz(dReLU_dz)
    """
    cases = [
        (np.array([[1.0], [0.0], [-5.0]]), np.array([[1.0], [0.0], [0.0]])),
        (
            np.array([[8.7], [-4.0], [-595.006], [0.0], [3.0]]),
            np.array([[1.0], [0.0], [0.0], [0.0], [1.0]]),
        ),
    ]
    run_test_suite(cases, fn)


def test_linear(module):
    """
    Tests Linear module: test_linear(Linear)
    """

    def nn_linear_forward(x):
        np.random.seed(0)
        linear = module(2, 3)
        return linear.forward(x)

    def nn_linear_forward_bias(x):
        np.random.seed(0)
        linear = module(2, 3)
        linear.W0 = np.array([[1], [1], [1]])
        return linear.forward(x)

    def nn_linear_backward(x):
        np.random.seed(0)
        linear = module(2, 3)
        linear.forward(x)
        dLdZ = np.array([[1, 1, 0, 0], [2, 0, 1, 0], [3, 0, 0, 1]])
        dLdA = linear.backward(dLdZ)
        return dLdA, linear.dLdW, linear.dLdW0

    def nn_linear_sgd(x):
        np.random.seed(0)
        linear = module(2, 3)
        linear.forward(x)
        dLdZ = np.array([[1, 1, 0, 0], [2, 0, 1, 0], [3, 0, 0, 1]])
        linear.backward(dLdZ)
        linear.sgd_step(0.005)
        return np.vstack([linear.W, linear.W0.T])

    X, y = super_simple_separable()

    print("***********************************")
    print("Test linear forward:")
    expected = np.array(
        [
            [
                10.417500637754383,
                6.911221682745654,
                20.733665048236965,
                22.891234399772113,
            ],
            [
                7.168722346625092,
                3.489987464919749,
                10.469962394759248,
                9.998261102396512,
            ],
            [
                -2.071054548689073,
                0.6941371647696142,
                2.0824114943088414,
                4.849668106971125,
            ],
        ]
    )
    run_test_suite([(X, expected)], nn_linear_forward)
    print("***********************************")

    print("Test linear forward w bias:")
    run_test_suite([(X, expected + 1)], nn_linear_forward_bias)
    print("***********************************")

    print("Test linear backward output:")
    expected = np.array(
        [
            [
                3.889497924054116,
                1.247373376201773,
                0.2829538755771419,
                0.6920722655660196,
            ],
            [
                2.1525571673658237,
                1.5845507770701677,
                1.3205629190941617,
                -0.6910398159642225,
            ],
        ]
    )
    run_test_suite([(X, expected)], lambda *args: nn_linear_backward(*args)[0])
    print("***********************************")

    print("Test linear backward updated dLdW:")
    expected = np.array([[5, 13, 18], [7, 16, 20]])
    run_test_suite([(X, expected)], lambda *args: nn_linear_backward(*args)[1])
    print("***********************************")

    print("Test linear backward updated dLdW0:")
    expected = np.array([[2], [3], [4]])
    run_test_suite([(X, expected)], lambda *args: nn_linear_backward(*args)[2])
    print("***********************************")

    print("Test linear sgd updated weights:")
    expected = np.array(
        [
            [1.222373376201773, 0.2179538755771419, 0.6020722655660197],
            [1.5495507770701678, 1.2405629190941616, -0.7910398159642225],
            [-0.01, -0.015, -0.02],
        ]
    )
    run_test_suite([(X, expected)], lambda *args: nn_linear_sgd(*args))
    print("***********************************")


def test_tanh(module):
    """
    Tests Tanh module: test_tanh(Tanh)
    """

    def nn_tanh_forward(x):
        np.random.seed(0)
        tanh = module()
        return tanh.forward(x)

    def nn_tanh_backward(x):
        np.random.seed(0)
        tanh = module()
        tanh.forward(x)
        dLdA = np.array([[1, 1, 0, 0], [2, 0, 1, 0]])
        return tanh.backward(dLdA)

    X, y = super_simple_separable()

    print("***********************************")
    print("Test tanh forward:")
    expected_forward = np.array(
        [
            [
                0.9640275800758169,
                0.9950547536867305,
                0.999999969540041,
                0.9999999999244973,
            ],
            [
                0.9999092042625951,
                0.9640275800758169,
                0.9999877116507956,
                0.9999092042625951,
            ],
        ]
    )
    run_test_suite([(X, expected_forward)], nn_tanh_forward)
    print("***********************************")

    print("Test tanh backward:")
    expected_backward = np.array(
        [
            [0.07065082485316443, 0.009866037165440211, 0.0, 0.0],
            [0.0003631664618877206, 0.0, 2.4576547405286142e-05, 0.0],
        ]
    )
    run_test_suite([(X, expected_backward)], nn_tanh_backward)
    print("***********************************")


def test_relu(module):
    """
    Tests ReLU module: test_relu(ReLU)
    """

    def nn_relu_forward(x):
        np.random.seed(0)
        relu = module()
        return relu.forward(x)

    def nn_relu_backward(x):
        np.random.seed(0)
        relu = module()
        relu.forward(x)
        dLdA = np.array([[1, 1, 0, 0], [2, 0, 1, 0]])
        return relu.backward(dLdA)

    X, y = super_simple_separable()

    print("***********************************")
    print("Test relu forward:")
    expected_forward = np.array([[2, 3, 9, 12], [5, 2, 6, 5]])
    run_test_suite([(X, expected_forward)], nn_relu_forward)
    print("***********************************")

    print("Test relu backward:")
    expected_backward = np.array([[1, 1, 0, 0], [2, 0, 1, 0]])
    run_test_suite([(X, expected_backward)], nn_relu_backward)
    print("***********************************")


def test_softmax(module):
    """
    Tests SoftMax module: test_softmax(SoftMax)
    """

    def nn_softmax_forward(x):
        np.random.seed(0)
        softmax = module()
        return softmax.forward(x)

    def nn_softmax_backward(x):
        np.random.seed(0)
        softmax = module()
        Ypred = softmax.forward(x)
        return softmax.class_fun(Ypred)

    X, y = super_simple_separable()

    print("***********************************")
    print("Test softmax forward:")
    expected_forward = np.array(
        [
            [
                0.04742587317756679,
                0.7310585786300048,
                0.9525741268224334,
                0.9990889488055993,
            ],
            [
                0.9525741268224333,
                0.2689414213699951,
                0.04742587317756678,
                0.0009110511944006454,
            ],
        ]
    )
    run_test_suite([(X, expected_forward)], nn_softmax_forward)
    print("***********************************")

    print("Test softmax class fun:")
    expected_backward = np.array([1, 0, 0, 0])
    run_test_suite([(X, expected_backward)], nn_softmax_backward)
    print("***********************************")


def test_nll(module):
    """
    Tests NLL module: test_nll(NLL)
    """
    y = np.array([[1, 0, 1, 0]])
    Y = for_softmax(y)
    ypred = np.array([[0.7, 0.3, 0.99, 0.99]])
    Ypred = for_softmax(ypred)

    def nn_nll_backward(_):
        nll = module()
        nll.forward(Ypred, Y)
        return nll.backward()

    print("***********************************")
    print("Test nll forward:")
    expected_forward = 5.328570409719057
    run_test_suite(
        [
            (
                None,
                expected_forward,
            )
        ],
        lambda _: module().forward(Ypred, Y),
    )
    print("***********************************")

    print("Test nll class fun:")
    expected_backward = np.array(
        [
            [0.30000000000000004, -0.30000000000000004, 0.010000000000000009, -0.99],
            [-0.30000000000000004, 0.3, -0.010000000000000009, 0.99],
        ]
    )
    run_test_suite(
        [
            (
                None,
                expected_backward,
            )
        ],
        nn_nll_backward,
    )
    print("***********************************")


def unit_test(name, expected, actual):
    if actual is None:
        print(name + ": unimplemented")
    elif np.allclose(expected, actual):
        print(name + ": OK")
    else:
        print(name + ": FAILED")
        print("expected: " + str(expected))
        print("but was: " + str(actual))


def test_sequential_components(nn, test_values):
    """Run one step of GD on a simple dataset with the specified
    network, and with batch size (b) = len(dataset), testing the
    behavior of each component of the neural network

    :param nn: A "Sequential" object representing a neural network

    :param test_values: A dictionary containing the expected values
    for the necessary unit tests

    """
    lrate = 0.005
    # data
    X, Y = super_simple_separable()

    # define the modules
    assert len(nn.modules) == 4
    linear_1, f_1, linear_2, f_2 = nn.modules
    Loss = nn.loss

    unit_test("linear_1.W", test_values["linear_1.W"], linear_1.W)
    unit_test("linear_1.W0", test_values["linear_1.W0"], linear_1.W0)
    unit_test("linear_2.W", test_values["linear_2.W"], linear_2.W)
    unit_test("linear_2.W0", test_values["linear_2.W0"], linear_2.W0)

    z_1 = linear_1.forward(X)
    unit_test("z_1", test_values["z_1"], z_1)
    a_1 = f_1.forward(z_1)
    unit_test("a_1", test_values["a_1"], a_1)
    z_2 = linear_2.forward(a_1)
    unit_test("z_2", test_values["z_2"], z_2)
    a_2 = f_2.forward(z_2)
    unit_test("a_2", test_values["a_2"], a_2)

    Ypred = a_2
    loss = Loss.forward(Ypred, Y)
    unit_test("loss", test_values["loss"], loss)
    dloss = Loss.backward()
    unit_test("dloss", test_values["dloss"], dloss)

    dL_dz2 = f_2.backward(dloss)
    unit_test("dL_dz2", test_values["dL_dz2"], dL_dz2)
    dL_da1 = linear_2.backward(dL_dz2)
    unit_test("dL_da1", test_values["dL_da1"], dL_da1)
    dL_dz1 = f_1.backward(dL_da1)
    unit_test("dL_dz1", test_values["dL_dz1"], dL_dz1)
    dL_dX = linear_1.backward(dL_dz1)
    unit_test("dL_dX", test_values["dL_dX"], dL_dX)

    linear_1.sgd_step(lrate)
    unit_test("updated_linear_1.W", test_values["updated_linear_1.W"], linear_1.W)
    unit_test("updated_linear_1.W0", test_values["updated_linear_1.W0"], linear_1.W0)
    linear_2.sgd_step(lrate)
    unit_test("updated_linear_2.W", test_values["updated_linear_2.W"], linear_2.W)
    unit_test("updated_linear_2.W0", test_values["updated_linear_2.W0"], linear_2.W0)


def test_sequential_sgd(nn, test_values):
    """Run one step of SGD on a simple dataset with the specified
    network

    :param nn: A "Sequential" object representing a neural network

    :param test_values: A dictionary containing the expected values
    for the necessary unit tests
    """
    lrate = 0.005
    # data
    X, Y = super_simple_separable()

    # define the modules
    assert len(nn.modules) == 4
    linear_1, f_1, linear_2, f_2 = nn.modules

    unit_test("linear_1.W", test_values["linear_1.W"], linear_1.W)
    unit_test("linear_1.W0", test_values["linear_1.W0"], linear_1.W0)
    unit_test("linear_2.W", test_values["linear_2.W"], linear_2.W)
    unit_test("linear_2.W0", test_values["linear_2.W0"], linear_2.W0)

    nn.sgd(X, Y, iters=1, lrate=lrate)

    unit_test("updated_linear_1.W", test_values["updated_linear_1.W"], linear_1.W)
    unit_test("updated_linear_1.W0", test_values["updated_linear_1.W0"], linear_1.W0)
    unit_test("updated_linear_2.W", test_values["updated_linear_2.W"], linear_2.W)
    unit_test("updated_linear_2.W0", test_values["updated_linear_2.W0"], linear_2.W0)


def test_sequential(sequential, linear, tanh, relu, softmax, nll):
    """
    Tests Sequential module: test_sequential(Sequential, Linear, Tanh, ReLU, SoftMax, NLL)
    """
    print("***********************************")
    print("Test for tanh activation and softmax output:")
    np.random.seed(0)
    test_sequential_components(
        sequential([linear(2, 3), tanh(), linear(3, 2), softmax()], nll()),
        test_1_values,
    )
    print("Test sgd:")
    np.random.seed(0)
    test_sequential_sgd(
        sequential([linear(2, 3), tanh(), linear(3, 2), softmax()], nll()),
        test_1_sgd_values,
    )
    print("***********************************")

    print("Test for relu activation and softmax output:")
    np.random.seed(0)
    test_sequential_components(
        sequential([linear(2, 3), relu(), linear(3, 2), softmax()], nll()),
        test_2_values,
    )
    print("Test sgd:")
    np.random.seed(0)
    test_sequential_sgd(
        sequential([linear(2, 3), relu(), linear(3, 2), softmax()], nll()),
        test_2_sgd_values,
    )
    print("***********************************")
