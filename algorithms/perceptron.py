import numpy as np


class Perceptron:
    def __init__(self):
        print("Created a Perceptron!")

    def fit(self, points, values, offset=False, epochs=10):
        if offset:
            points = np.c_[np.ones(points.shape[0]), points]
        print("Initial Points:", points)
        para = np.zeros(points.shape[1])
        # para = np.array([-3,-3,3])
        print("Initial Parameters:", para)
        for epoch in range(epochs):
            print("\nEpoch: ", epoch)
            for p, v in zip(points, values):
                predictions = np.dot(para, p)
                print("Applying on", p, para, "=", predictions, "with value", v, end="")
                if v * predictions <= 0:
                    para = para + v * p
                print("; Updated parameters:", para)


if __name__ == "__main__":
    x = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]])
    y = np.array([-1, 1, -1])
    m_perc = Perceptron()
    m_perc.fit(x, y, offset=False, epochs=50)
