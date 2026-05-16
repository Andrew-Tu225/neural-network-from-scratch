# Neural Network From Scratch

This project is a small neural network implementation built from scratch with NumPy and trained on the MNIST handwritten digit dataset.

The goal is to understand the mechanics behind a basic neural network without relying on deep learning frameworks like TensorFlow or PyTorch. The notebook walks through loading MNIST, preprocessing the data, implementing forward propagation, computing gradients with backpropagation, and training the model with gradient descent.

## What This Project Does

The model is a simple two-layer neural network for classifying handwritten digits from `0` to `9`.

It uses:

- MNIST images with 784 input features, one for each pixel in a `28 x 28` image
- One hidden layer with ReLU activation
- One output layer with softmax activation
- One-hot encoded labels
- Cross-entropy-style softmax gradient
- Batch gradient descent
- Accuracy tracking during training

The project is intentionally educational. Most of the important neural network pieces are implemented manually, including:

- Parameter initialization
- ReLU activation
- Softmax activation
- Forward propagation
- One-hot label conversion
- Backpropagation
- Weight and bias updates
- Prediction and accuracy calculation

## Dataset

This project uses the MNIST dataset from OpenML through scikit-learn:

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=True, parser='auto')
```

MNIST contains grayscale images of handwritten digits. Each image is flattened into 784 pixel values, and each label identifies the digit shown in the image.

The notebook splits the data into:

- `60,000` training examples
- `10,000` testing examples

Pixel values are normalized from the original `0` to `255` range into the `0` to `1` range before training.

## Project Structure

```text
.
├── main.ipynb
└── README.md
```

`main.ipynb` contains the full implementation and training workflow.

## How to Run It Yourself

### 1. Clone or Download the Project

If you are using Git:

```bash
git clone https://github.com/Andrew-Tu225/neural-network-from-scratch
```

Or download the project files and open the folder locally.

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn notebook
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```text
main.ipynb
```

Run the notebook cells from top to bottom.

The first time you run it, scikit-learn will download MNIST from OpenML, so you will need an internet connection.

## Training

The main training call looks like this:

```python
W1, B1, W2, B2 = gradient_descent(X_train, Y_train, 1000, 0.1)
```

This trains the network for `1000` iterations with a learning rate of `0.1`.

During training, the notebook prints accuracy every 20 iterations so you can see whether the network is improving.

## Key Learning Notes

One important detail in this project is making sure the gradient is averaged over the number of training examples.

After transposing the training data, the shape is:

```python
X_train.shape == (784, 60000)
```

That means the number of examples is the second dimension:

```python
n_x, m = X_train.shape
```

Using the wrong value for `m` can make gradient updates far too large, which can cause the network to stop learning.