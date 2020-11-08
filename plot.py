import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation


last_run_folder = os.path.join(os.getcwd(), "last_run")

time = np.load(os.path.join(last_run_folder, "time.npy"))
temp = np.load(os.path.join(last_run_folder, "temp.npy"))
u = np.load(os.path.join(last_run_folder, "u.npy"))
actions = np.load(os.path.join(last_run_folder, "actions.npy"))
nusselt = np.load(os.path.join(last_run_folder, "nusselt.npy"))

print(actions[0].sum())
sys.exit()

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

        plt.title(f"Time: {time[i]:.1f}s nusselt: {nusselt[i]:.3f}")

#plot
anim = animation.FuncAnimation(fig, animate, frames = len(temp))
plt.show()
