from dataset import x1, x2
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.scatter(x1, x2)

ax.set_xlabel(r"$x1$", fontsize=15)
ax.set_ylabel(r"$x2$", fontsize=15)
ax.set_title("Scatter plot")

ax.grid(True)

plt.show()
