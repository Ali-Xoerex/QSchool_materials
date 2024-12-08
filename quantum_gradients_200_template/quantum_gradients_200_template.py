#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #
    no_execs = 1
    gradient_memo = {}
    s_h = (np.pi/2)
    s_g = (np.pi/2)
    f_theta = circuit(weights)
    # The gradient calculations
    # saving intermediate calculations along the way
    for i in range(weights.shape[0]):
        w_copy = np.copy(weights)
        w_copy[i] += s_g
        first_term = circuit(w_copy)
        no_execs += 1
        w_copy = np.copy(weights)
        w_copy[i] -= s_g
        second_term = circuit(w_copy)
        no_execs += 1
        gradient[i] = (first_term - second_term) / 2*np.sin(s_g)
        gradient_memo[i] = [first_term,second_term]
    # calculating the hessian
    # and saving calculations along the way
    for i in range(weights.shape[0]):
        for j in range(weights.shape[0]):
            if i == j: # diagonal elements. can use gradient terms
                hessian[i][j] += (gradient_memo[i][0] + gradient_memo[i][1]-2*f_theta) / 2
            elif i < j:
                w_copy = np.copy(weights)
                w_copy[i] += s_h
                w_copy[j] += s_h
                first_term = circuit(w_copy)
                no_execs += 1
                w_copy = np.copy(weights)
                w_copy[i] -= s_h
                w_copy[j] -= s_h
                second_term = circuit(w_copy)
                no_execs += 1
                w_copy = np.copy(weights)
                w_copy[i] += s_h
                w_copy[j] -= s_h
                third_term = circuit(w_copy)
                no_execs += 1
                w_copy = np.copy(weights)
                w_copy[i] -= s_h
                w_copy[j] += s_h
                fourth_term = circuit(w_copy)
                no_execs += 1
                hessian[i][j] += (first_term + second_term - third_term - fourth_term) / 4 * (np.sin(s_h) ** 2)
                hessian[j][i] += (first_term + second_term - third_term - fourth_term) / 4 * (np.sin(s_h) ** 2)
    # QHACK #
    return gradient, hessian,no_execs


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, executes = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        executes,
        sep=","
    )
