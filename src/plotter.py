import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, title="Plot", xlabel="X-axis", ylabel="Y-axis"):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot(self, x, y, label=None):
        plt.figure()
        plt.plot(x, y, label=label)
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        if label:
            plt.legend()
        plt.grid()
        plt.show()
