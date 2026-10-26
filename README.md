# Perceptron

A 2018 learning exercise: a single artificial neuron, written in Python with NumPy, that learns a tiny binary classification task by gradient descent.

This is a finished exercise kept as a record of early learning. It is not maintained.

## What it does

The whole program is one script, [`Docs/neural.py`](Docs/neural.py) (it lives in `Docs/` despite the folder name). It:

1. Defines 9 training examples, each with 10 binary features and a binary label.
2. Initializes 10 weights uniformly in [-1, 1) with a fixed seed (`np.random.seed(1)`), so every run prints the same numbers.
3. Runs 20,000 full-batch updates: compute `sigmoid(X @ w)`, multiply the error by the slope of the sigmoid, then update `w += X.T @ delta`. There is no bias term and no explicit learning rate.
4. Prints the training data, then the neuron's output for one unseen input, `[1, 1, 0, 1, 1, 1, 1, 1, 1, 1]`.

In this dataset the label is always equal to the third feature (index 2), so the neuron learns to copy that feature: its weight ends near 10, while the other weights stay small. The test input has a 0 in that position, so the prediction is close to 0.

### A note on the name

Despite the repository name, this is not a classic perceptron, which uses a step function and the perceptron learning rule. It is a single sigmoid neuron, the same model as logistic regression. The update rule follows the squared-error gradient, as in the tutorial below, rather than the cross-entropy loss usually paired with logistic regression.

## Credit

The code follows Andrew Trask's tutorial [A Neural Network in 11 lines of Python (Part 1)](https://iamtrask.github.io/2015/07/12/basic-python-network/) (2015): the `nonlin(x, deriv)` helper, the seeded `2*np.random.random(...)-1` initialization, the training loop and its comment come from there. The dataset is different from the tutorial's (9 examples with 10 features instead of 4 with 3).

## Running it

Requires Python 3 and NumPy. Checked with Python 3.13 and NumPy 2.5.4, and with Python 3.9 and NumPy 2.0.2.

```sh
git clone https://github.com/MSJawad/perceptron.git
cd perceptron
python3 -m venv .venv
source .venv/bin/activate
pip install numpy
python Docs/neural.py
```

## Output

The script prints the training inputs, the training labels, and then the prediction for the test input:

```text
[[0 0 0 0 0 0 0 0 0 0]
 [0 1 0 0 1 1 0 1 0 0]
 [1 0 1 1 0 1 1 1 1 1]
 [1 0 1 0 1 0 1 1 0 0]
 [1 0 0 0 1 1 0 0 1 1]
 [1 0 0 1 0 1 0 1 1 1]
 [0 1 1 0 1 1 0 1 0 0]
 [1 0 1 0 0 1 0 0 0 1]
 [1 1 0 1 0 0 1 1 0 0]]
[[0]
 [0]
 [1]
 [1]
 [0]
 [0]
 [1]
 [1]
 [0]]
[0.00010538]
```

The last line is the predicted probability that the test input belongs to class 1, about 0.0001.

## Author

Mohammad Solaiman Jawad

## License

MIT. See [LICENSE](LICENSE).
