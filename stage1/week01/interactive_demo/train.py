import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

x = rng.uniform(-2, 2, size=100)
noise = rng.normal(0, 0.15, size=100)
y = 2 * x + 1 + noise


def train(lr, epochs=100):
    w = 0.0
    b = 0.0

    losses = [np.mean((w * x + b - y) ** 2)]

    for epoch in range(epochs):
        y_pred = w * x + b

        error = y_pred - y

        dw = 2 * np.mean(error * x)
        db = 2 * np.mean(error)

        w -= lr * dw
        b -= lr * db

        loss = np.mean((w * x + b - y) ** 2)
        losses.append(loss)

    return losses, w, b


learning_rates = [0.005, 0.02, 0.1]

for lr in learning_rates:
    losses, w, b = train(lr)

    plt.plot(
        range(len(losses)),
        losses,
        label=f"lr={lr}"
    )

    print(f"lr={lr}, loss={losses[-1]:.6f}")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Learning Rate Comparison")
plt.legend()
plt.grid(True)
plt.show()