#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Mini neural network (pure numpy, written from scratch) — build a classifier
without any deep learning framework
=============================================================================

Why not use an off-the-shelf framework (PyTorch, etc.)?

Two reasons:
1. It isn't installed on your machine, and installing it takes hundreds of MB.
   This task (classifying 96 numbers into 3 sleep postures) doesn't need
   anything that heavy.
2. Writing it by hand is the only way to truly understand "what a neural
   network actually does". Frameworks hide the magic; a hand-written version
   lays out every step in front of you. When you later use PyTorch, you'll
   know what's going on under the hood.

What does this network look like?

    input (96 pressure values) → hidden (32 neurons, ReLU) → output (3 posture scores) → probabilities

Every connection has a "weight" — the weight is the "experience" the network
learned. Training is nothing but repeatedly nudging those weights so the
predictions get more accurate.

The training principle, in one sentence:

    Forward pass: data flows forward, produce a prediction → measure "how far off" (loss)
    Backward pass: work backward, compute how much each weight is to blame (gradient)
    Update: move each weight one small step in the direction that reduces the error

Repeat a few thousand times and the network "learns". That's all there is.

=============================================================================
"""

import numpy as np


def relu(x):
    """ReLU activation: negatives are zeroed out, positives pass through unchanged.
    Mimics real neurons that don't fire until sufficiently stimulated. Without
    it, no matter how deep the network is it collapses to a single layer."""
    return np.maximum(0, x)


def softmax(z):
    """Turn arbitrary scores into probabilities that sum to 1.
    E.g. [3.0, 1.0, 0.1] → [0.84, 0.11, 0.04], meaning 84% confidence in the
    first class. Subtracting the max is a numerical safeguard: it prevents exp
    overflow without changing the result."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TinyMLP:
    """Two-layer fully-connected network: input → hidden (ReLU) → output (softmax)."""

    def __init__(self, n_input, n_hidden, n_output, seed=0):
        """
        Initialize the weights. They start as small random numbers — a newborn
        network knows nothing and just guesses. Training slowly tunes these
        numbers into "sensible" values.

        Multiplying by 0.5/sqrt(n_input) is an initialization trick: it keeps
        the initial outputs from being too large or too small.
        """
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5, size=(n_input, n_hidden)) / np.sqrt(n_input)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 0.5, size=(n_hidden, n_output)) / np.sqrt(n_hidden)
        self.b2 = np.zeros(n_output)

    # ------------------------------------------------------------------
    # Forward pass: data flows forward
    # ------------------------------------------------------------------
    def forward(self, X):
        """
        X has shape (n_samples, 96).

        Returns (probabilities, cache). The cache stores intermediate results
        needed by the backward pass — like the scratch paper you keep when
        solving a math problem so you can review it while checking your answer.
        """
        z1 = X @ self.W1 + self.b1          # weighted sum for each hidden neuron
        a1 = relu(z1)                        # apply the activation
        z2 = a1 @ self.W2 + self.b2          # weighted sum for the output layer
        probs = softmax(z2)                  # scores → probabilities
        return probs, (X, z1, a1)

    # ------------------------------------------------------------------
    # Backward pass: work backward to compute "how much each weight is to blame"
    # ------------------------------------------------------------------
    def backward(self, cache, y_onehot):
        """
        y_onehot is the one-hot encoding of the ground-truth answer.
        For example, if the correct class is "side" (index 1), three classes
        are written as [0, 1, 0].

        The core formulas come from the chain rule of calculus. The nicest step:
        the gradient of softmax + cross-entropy equals exactly (predicted prob -
        true label). Intuitively: the more wrong the prediction, the bigger the
        penalty.
        """
        X, z1, a1 = cache
        n = X.shape[0]

        # Output-layer "responsibility": predicted probability - ground truth
        dz2 = (self._last_probs - y_onehot) / n
        # Split that responsibility among each W2 weight, weighted by the input (hidden output)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)

        # Propagate the responsibility back to the hidden layer
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)                 # ReLU gradient: pass positives through, zero negatives
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        return dW1, db1, dW2, db2

    # ------------------------------------------------------------------
    # One training step: forward → compute loss → backward → update
    # ------------------------------------------------------------------
    def train_step(self, X, y, lr=0.5):
        """
        Run the full "forward → backward → update" cycle and return the average
        loss for this step.

        lr is the learning rate: how big a step each weight takes per update.
        Too big → overshoots and oscillates; too small → learns slowly.
        0.5 works well for this task.
        """
        n = len(X)
        probs, cache = self.forward(X)
        self._last_probs = probs             # used in backward (see above)

        # Cross-entropy loss: only look at the probability of the correct class.
        # Predicted correctly (prob → 1) → loss → 0; wrong (prob → 0) → loss → ∞.
        eps = 1e-12
        loss = -np.log(probs[np.arange(n), y] + eps).mean()

        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(n), y] = 1.0
        dW1, db1, dW2, db2 = self.backward(cache, y_onehot)

        # Gradient descent: move each weight by lr in the direction of the negative gradient
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

        return float(loss)

    # ------------------------------------------------------------------
    # Prediction: how the trained network is used
    # ------------------------------------------------------------------
    def predict(self, X):
        """Return the predicted class index (argmax probability) for each sample."""
        probs, _ = self.forward(X)
        return probs.argmax(axis=1)

    def accuracy(self, X, y):
        """Score on X: the fraction of predictions that are correct."""
        return float((self.predict(X) == y).mean())
