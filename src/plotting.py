import matplotlib.pyplot as plt
import random
import numpy as np
import time


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
            edgecolor="black",
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
            xlim=self.x_limits, ylim=self.y_limits, xlabel="X", ylabel="Y", zlabel="U"
        )

    def plot2d_contour(
        self, color_map="coolwarm", contour_levels=15, linestyle="dashed"
    ):
        """
        This function creates a 2D contour plot of the real and predicted u-values.
        """
        fig, ((real, pred), (diff, none)) = plt.subplots(
            2, 2, figsize=(12, 12))
        fig.suptitle("Real and Predicted U-Values")

        real.contour(
            self.test_x,
            self.test_y,
            self.true_u,
            levels=contour_levels,
            linewidths=1,
            linestyle=linestyle,
            colors=["black"],
            alpha=1,
        )
        realc = real.contourf(
            self.x,
            self.y,
            self.true_u,
            levels=100,
            cmap=color_map,
            alpha=0.6,
        )
        real.legend(["Real"])
        fig.colorbar(realc, ax=real)

        pred.contour(
            self.test_x,
            self.test_y,
            self.pred_u,
            levels=contour_levels,
            linewidths=1,
            linestyle=linestyle,
            colors=["black"],
            alpha=1,
        )
        predc = pred.contourf(
            self.x,
            self.y,
            self.pred_u,
            levels=100,
            cmap=color_map,
            alpha=0.6,
        )
        pred.legend(["Predicted"])
        fig.colorbar(predc, ax=pred)

        real.set(xlim=self.x_limits, ylim=self.y_limits,
                 xlabel="X", ylabel="Y")
        pred.set(
            title="Predicted",
            xlim=self.x_limits,
            ylim=self.y_limits,
            xlabel="X",
            ylabel="Y")

        diff.contour(
            self.test_x,
            self.test_y,
            self.pred_u - self.true_u,
            levels=contour_levels,
            linewidths=1,
            linestyle=linestyle,
            colors=["black"],
            alpha=1,
        )
        diffc = diff.contourf(
            self.x,
            self.y,
            self.pred_u - self.true_u,
            levels=100,
            cmap=color_map,
            alpha=0.6,
        )
        diff.set(
            title="Real - Predicted",
            xlim=self.x_limits,
            ylim=self.y_limits,
            xlabel="X",
            ylabel="Y")
        real.set(title="Real")
        diff.legend(["Real - Predicted"])
        fig.colorbar(diffc, ax=diff)

    def plot2d_fix_x(self, x_i=None):
        """
        This function creates a 2D plot by fixing the x-coordinate at a specific value.
        If no value is provided for x_i, the function randomly selects an index within the valid range.
        """
        if x_i == None:
            x_i = random.randint(0, len(self.x[0]) - 1)
        fig1 = plt.figure()
        flat = fig1.add_subplot()
        flat.plot(
            np.linspace(self.y_limits[0],
                        self.y_limits[1], len(self.pred_u[x_i])),
            self.pred_u[x_i],
            label="pred",
        )
        flat.plot(
            np.linspace(self.y_limits[0],
                        self.y_limits[1], len(self.true_u[x_i])),
            self.true_u[x_i],
            label="real",
        )
        flat.legend(fontsize=15)
        flat.set(
            title=f"U-vales with fixed x={round(self.x[0, x_i], 2)}",
            xlabel="Y",
            ylabel="U",
        )


if __name__ == ("__main__"):

    def real_u(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    grid_size = (50, 50)
    plotting_grid_size = (200, 200)
    x_limits = (0, 2)
    y_limits = (0, 2)
    border_grid_size = (50, 50)
    x = np.linspace(x_limits[0], x_limits[1], plotting_grid_size[0])
    y = np.linspace(y_limits[0], y_limits[1], plotting_grid_size[1])
    x, y = np.meshgrid(x, y)
    test_x = np.linspace(x_limits[0], x_limits[1], plotting_grid_size[0])
    test_y = np.linspace(x_limits[0], x_limits[1], plotting_grid_size[1])
    test_x, test_y = np.meshgrid(test_x, test_y)
    train_u = real_u(x, y)
    true_u = real_u(test_x, test_y)
    test_coord = np.column_stack((test_x.flatten(), test_y.flatten()))
    pred_coord = list()
    pred_u = true_u
    plotter = NNPlots(test_x, test_y, true_u, x, y, pred_u, x_limits, y_limits)
    plotter.plot2d_contour()
