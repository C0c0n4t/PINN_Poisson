import matplotlib.pyplot as plt
import random
import numpy as np


class NNPlots:
    def __init__(self, test_x, test_y, true_u, x, y, pred_u, x_limits, y_limits):
        self.test_x = test_x
        self.test_y = test_y
        self.true_u = true_u

        self.x = x
        self.y = y
        self.pred_u = pred_u

        self.x_limits = x_limits
        self.y_limits = y_limits

    @staticmethod
    def plotLoss(train_loss):
        plt.figure(figsize=(10, 8))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.plot(train_loss)
        plt.show()

    def plot3d(self):
        """
        3d plot with 2 surfaces: real and predicted
        """
        ax = plt.figure().add_subplot(projection="3d")
        ax.plot_surface(
            self.test_x,
            self.test_y,
            self.true_u,
            edgecolor="royalblue",
            lw=0.5,
            rstride=8,
            cstride=8,
            alpha=0.4,
            # cmap="BuGn",
            facecolor="red",
            label="real",
        )
        ax.plot_surface(
            self.x,
            self.y,
            self.pred_u,
            edgecolor="royalblue",
            lw=0.5,
            rstride=8,
            cstride=8,
            alpha=0.4,
            # cmap="plasma",
            facecolor="blue",
            label="pred",
        )
        # ax.scatter(x, y, train_u, s=0.5, label="train")
        ax.legend(fontsize=15)
        ax.set(
            xlim=self.x_limits, ylim=self.y_limits, xlabel="X", ylabel="Y", zlabel="Z"
        )

    def plot2d_fix_x(self, x_i=None):
        """
        This function creates a 2D plot by fixing the x-coordinate at a specific value.
        If no value is provided for x_i, the function randomly selects an index within the valid range.
        """
        if x_i == None:
            x_i = random.randint(0, self.grid_size[0] - 1)
        fig1 = plt.figure()
        flat = fig1.add_subplot()
        flat.plot(
            np.linspace(self.y_limits[0], self.y_limits[1], len(self.pred_u[x_i])),
            self.pred_u[x_i],
            label="pred",
        )
        flat.plot(
            np.linspace(self.y_limits[0], self.y_limits[1], len(self.true_u[x_i])),
            self.true_u[x_i],
            label="real",
        )
        flat.legend(fontsize=15)
