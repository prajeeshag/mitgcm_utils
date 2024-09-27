import matplotlib.pyplot as plt
import numpy as np

var_names = ["T_E"]
nx = 780
ny = 60
suffix = ""
for name in var_names:
    var_tmp = np.fromfile(f"{name}.bin", ">f4")
    nt = int(var_tmp.shape[0] / nx / ny)
    var = var_tmp.reshape([nt, ny, nx])
    # get a random integer between 0 to nt-1
    # t = np.random.randint(0, nt)
    t = 0
    # plot pcolormesh of the field and save the file
    plt.pcolormesh(var[t, :, :])  # type: ignore
    # reverse y-axis
    plt.gca().invert_yaxis()
    plt.colorbar()  # type: ignore
    plt.title(f"{name} at time step {t}")  # type: ignore
    plt.savefig(f"{name}{suffix}.png")  # type: ignore
    plt.clf()
