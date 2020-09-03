import sys

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation


time = np.load("time.npy")
temp = np.load("temp.npy")
u = np.load("u.npy")

msg = "usage: python3 plot.py u/T"

assert len(sys.argv) == 2, msg
assert sys.argv[1] in ["u", "T"], msg

fig, ax = plt.subplots()
N = len(time)

if sys.argv[1] == "u":
    def animate(i):
        ax.quiver(u[i, 0], u[i, 1])
        plt.title(f"Time: {time[i]:.1f}s")

elif sys.argv[1] == "T":
    COLORBAR = False

    def animate(i):
        global COLORBAR

        cs = ax.contourf(temp[i, :, :])
        if not COLORBAR:
            COLORBAR = True
            fig.colorbar(cs)

        plt.title(f"Time: {time[i]:.1f}s")

#plot
anim = animation.FuncAnimation(fig, animate, frames = len(temp))
plt.show()
