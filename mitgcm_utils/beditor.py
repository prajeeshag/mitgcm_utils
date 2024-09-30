import logging
import math
from enum import Enum
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button
from mpl_interactions import panhandler, zoom_factory  # type: ignore
from scipy import ndimage  # type: ignore

from .utils import load_bathy

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)

app = typer.Typer()


@app.command()
def clip_bathy(
    bathy_file: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        writable=True,
        dir_okay=False,
        file_okay=True,
        help="MITgcm bathymetry file",
    ),
    mindepth: float = typer.Option(
        -5,
        help="Minimum depth (-ve downwards), z[z > mindepth]=landvalue",
    ),
    maxdepth: float = typer.Option(
        None,
        help="Maximum depth (-ve downwards), z[z < maxdepth]= maxdepth",
    ),
    landvalue: float = typer.Option(
        100.0,
        help="Land value",
    ),
):
    """
    Clip the bathymetry of MITgcm between min and max values.
    """
    if not any([mindepth, maxdepth]):
        raise typer.BadParameter("Neither mindepth or maxdepth is given.")

    output = bathy_file

    logger.info(f"Reading bathymetry from {bathy_file}")
    z = np.fromfile(bathy_file, ">f4")
    logger.info("Clipping bathymetry")
    _clip_bathy(z, mindepth, maxdepth, landvalue)
    logger.info(f"Saving bathymetry to file {output}")
    z.astype(">f4").tofile(output)


def _clip_bathy(z, mindepth=None, maxdepth=None, landvalue=100.0):  # type: ignore
    if mindepth:
        z[z > mindepth] = landvalue
    if maxdepth:
        z[z < maxdepth] = maxdepth


class FeatureName(Enum):
    ISLAND = "island"
    POND = "POND"
    CREEK = "CREEK"


class Features:
    def __init__(self, name: FeatureName, bathy_file: Path, nx: int, ny: int) -> None:
        self.name = name
        self.out_file = bathy_file
        logger.info(f"Reading bathymetry from {self.out_file}")
        self.z = load_bathy(bathy_file, nx, ny)  # type: ignore
        self.min_depth = np.amax(self.z[self.z < 0])
        logger.info(f"Minimum ocean depth: {self.min_depth}")
        self.min_height = 100.0

        if name == FeatureName.ISLAND:
            self.mask = np.where(self.z >= 0, 1, 0)
            self.cval = self.min_depth
        elif name == FeatureName.POND:
            self.mask = np.where(self.z < 0, 1, 0)
            self.cval = self.min_height
        elif name == FeatureName.CREEK:
            mask = np.where(self.z < 0, 1, 0)
            self._creek_mask(mask)  # type: ignore
            self.cval = self.min_height
        else:
            raise NotImplementedError(f"Allowed feature names are {FeatureName}")

        self.shown = False
        self.boxes = []
        self._get_features()

    def _creek_mask(self, mask, n_neibhours=3):  # type: ignore
        cmask = np.zeros(mask.shape, dtype=int)  # type: ignore
        for y in range(mask.shape[0]):  # type: ignore
            for x in range(mask.shape[1]):  # type: ignore
                # Check if the pixel is black (i.e., part of a feature)
                if mask[y, x] == 1:
                    y1 = max(y - 1, 0)
                    y2 = min(y + 2, mask.shape[0] + 1)  # type: ignore
                    x1 = max(x - 1, 0)
                    x2 = min(x + 2, mask.shape[1] + 1)  # type: ignore
                    # Count the number of neighbors of the pixel
                    # num_neighbors = np.sum(mask[y1:y2, x1:x2]) - 1
                    nyn = min(np.sum(mask[y1:y2, x]) - 1, 1)  # type: ignore
                    nxn = min(np.sum(mask[y, x1:x2]) - 1, 1)  # type: ignore

                    # If the pixel has one or two neighbors, label it
                    if nxn + nyn <= 1:
                        cmask[y, x] = 1
        self.mask = cmask

    def _get_features(self):
        self.array, num_labels = ndimage.label(self.mask)  # type: ignore
        self.labels = list(range(1, num_labels + 1))

    def max_points(self, n, array=None):  # type: ignore
        labels = []
        if array is None:
            array = self.array.copy()
        for i in self.labels:
            npoints = np.count_nonzero(array == i)  # type: ignore
            if npoints > 0 and npoints <= n:
                labels.append(i)  # type: ignore
            else:
                array[array == i] = 0
        return array, labels

    def no_edge(self, array=None):
        if array is None:
            labeled_array = self.array
        labels = self.labels
        edge_pixels = np.concatenate(  # type: ignore
            (
                labeled_array[0, :],  # type: ignore
                labeled_array[-1, :],  # type: ignore
                labeled_array[:, 0],  # type: ignore
                labeled_array[:, -1],  # type: ignore
            )  # type: ignore
        )
        # get the labels that touch the edges
        edge_labels = np.intersect1d(labels, edge_pixels)  # type: ignore
        # get the labels that do not touch the edges
        pool_labels = np.setdiff1d(labels, edge_labels)

        pools = labeled_array  # type: ignore
        for i in edge_labels:
            pools[pools == i] = 0
        return pools, pool_labels

    def get_boxes(self, array, labels):  # type: ignore
        boxes = []
        for i in labels:
            points = np.where(array == i)  # type: ignore
            ny, nx = array.shape  # type: ignore
            x = np.max([np.min(points[1]) - 1, 0])  # type: ignore
            x2 = np.min([np.max(points[1]) + 1, nx])  # type: ignore

            y = np.max([np.min(points[0]) - 1, 0])  # type: ignore
            y2 = np.min([np.max(points[0]) + 1, ny])  # type: ignore

            width = x2 - x + 1  # type: ignore
            height = y2 - y + 1  # type: ignore

            # recalibrate rectangle if it is too small
            rwidth = np.max([width, nx // 100])  # type: ignore
            rheight = np.max([height, ny // 100])  # type: ignore
            x = x - (rwidth - width) // 2
            y = y - (rheight - height) // 2
            width = rwidth  # type: ignore
            height = rheight  # type: ignore

            boxes.append(  # type: ignore
                patches.Rectangle(
                    (x, y), width, height, linewidth=1, edgecolor="k", facecolor="none"  # type: ignore
                )
            )
        return boxes  # type: ignore

    def nn_average(self, _gx, _gy, avgtype="normal") -> bool:  # type: ignore
        logger.info(f"Averaging ({avgtype}) for point {(_gy,_gx)}")
        ny, nx = self.z.shape
        y1, y2 = max(_gy - 1, 0), min(_gy + 2, ny + 1)  # type: ignore
        x1, x2 = max(_gx - 1, 0), min(_gx + 2, nx + 1)  # type: ignore
        _z = self.z[y1:y2, x1:x2]
        if avgtype == "deepen":
            _z = _z[_z < self.z[_gy, _gx]]
        else:
            _z = _z[_z < 0.0]
        if len(_z) < 1:
            return False
        self.z[_gy, _gx] = np.mean(_z)
        return True

    def edit_features(self, n_points: int = 1, min_depth=None):  # type: ignore
        """Edit features"""

        def delete_islands(event):  # type: ignore
            logger.info(f"Deleting {self.name}")

            if self.name == FeatureName.ISLAND:
                all_done = False
                while not all_done:
                    all_done = True
                    for label in labels:  # type: ignore
                        idx = zip(*np.where(array == label))  # type: ignore
                        for j, i in idx:
                            if self.z[j, i] < 0.0:  # if ocean
                                continue
                            all_done = all_done and self.nn_average(i, j)  # type: ignore
            else:
                for label in labels:  # type: ignore
                    self.z[array == label] = self.cval
            logger.info(f"Saving bathymetry to file {self.out_file}")
            self.z.astype(">f4").tofile(self.out_file)
            plt.close()

        levels = list(np.linspace(-3000, -200, 10))[:-1] + list(
            np.linspace(-200, 0, 21)
        )
        levels = [-0.0000001 if item == 0.0 else item for item in levels]
        cmap = plt.cm.jet  # type: ignore
        cmap.set_over("white")  # type: ignore
        norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)
        with plt.ioff():
            fig, ax = plt.subplots()
        mesh = ax.pcolormesh(self.z, cmap=cmap, norm=norm)
        ax.set_aspect("equal")
        array, labels = self.max_points(n_points)
        boxes = self.get_boxes(array, labels)
        nf = len(boxes)
        logger.info(f"Number of {self.name} with grid points <= {n_points}: {nf} ")
        if nf == 0:
            logger.info(f"No {self.name} detected with grid points <= {n_points}")
            return

        for rect in boxes:
            ax.add_patch(rect)
        fig.canvas.draw()

        fig.subplots_adjust(bottom=0.2)
        # axnext = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        axdel = fig.add_axes([0.81, 0.05, 0.1, 0.075])
        bndel = Button(axdel, "Delete")
        bndel.on_clicked(delete_islands)
        plt.colorbar(mesh)
        plt.show()


@app.command("del_islands")
def del_islands(
    bathy_file: Path = typer.Option(..., file_okay=True, readable=True, dir_okay=False),
    nx: int = typer.Option(...),
    ny: int = typer.Option(...),
    n_points: int = typer.Option(
        1,
        help="Only consider features having grid points <= `n_points`",
    ),
):
    Features(FeatureName.ISLAND, bathy_file, nx, ny).edit_features(n_points=n_points)


@app.command("del_ponds")
def del_ponds(
    bathy_file: Path = typer.Option(..., file_okay=True, readable=True, dir_okay=False),
    nx: int = typer.Option(...),
    ny: int = typer.Option(...),
    n_points: int = typer.Option(
        1,
        help="Only consider features having grid points <= `n_points`",
    ),
):
    Features(FeatureName.POND, bathy_file, nx, ny).edit_features(n_points=n_points)


@app.command("del_creeks")
def del_creeks(
    bathy_file: Path = typer.Option(..., file_okay=True, readable=True, dir_okay=False),
    nx: int = typer.Option(...),
    ny: int = typer.Option(...),
    n_points: int = typer.Option(
        1,
        help="Only consider features having grid points <= `n_points`",
    ),
):
    Features(FeatureName.CREEK, bathy_file, nx, ny).edit_features(n_points=n_points)


def remove_plt_keymaps():
    for item in plt.rcParams:
        if item.startswith("keymap."):
            for i in plt.rcParams[item]:
                plt.rcParams[item].remove(i)
                # print(f"{item} - {i}")


class EditBathy:
    def __init__(self, bathy_file, nx, ny) -> None:
        remove_plt_keymaps()
        self.out_file = bathy_file
        logger.info(f"Reading bathymetry from {self.out_file}")
        self.z = load_bathy(bathy_file, nx, ny)
        self.z[self.z >= 0] = 1000.0
        self.min_depth = np.amax(self.z[self.z < 0])
        logger.info(f"Minimum ocean depth {self.min_depth}")
        self.lnd_val = 100.0
        levels = list(np.linspace(-3000, -200, 10))[:-1] + list(
            np.linspace(-200, 0, 21)
        )
        levels = [-0.0000001 if item == 0.0 else item for item in levels]
        cmap = plt.cm.jet
        cmap.set_over("white")
        norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)
        with plt.ioff():
            self.fig, self.ax = plt.subplots()
        self.mesh = self.ax.pcolormesh(self.z, cmap=cmap, norm=norm, picker=True)
        self.ax.set_aspect("equal")
        plt.colorbar(self.mesh)

        # self.mesh = self.ax.pcolormesh(pools, cmap=self.cmap, norm=self.norm)
        self.fig.canvas.mpl_connect("button_press_event", self._on_pick)
        self.fig.subplots_adjust(bottom=0.2)
        self.axsave = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        self.bndel = Button(self.axsave, "Save")
        self.bndel.on_clicked(self.save_z)

        _ = zoom_factory(self.ax)
        _ = panhandler(self.fig, button=2)
        plt.show()

    def save_z(self, event):
        logger.info(f"Saving to file {self.out_file}")
        self.z.astype(">f4").tofile(self.out_file)

    def _on_pick(self, event):
        logger.debug("_on_pick")
        mouseevent = event
        key = event.key

        if mouseevent.xdata is None or mouseevent.ydata is None:
            return

        logger.debug(
            "%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f, key=%s"
            % (
                "double" if mouseevent.dblclick else "single",
                mouseevent.button,
                mouseevent.x,
                mouseevent.y,
                mouseevent.xdata,
                mouseevent.ydata,
                key,
            )
        )
        if key is None:
            return

        if mouseevent.button is not MouseButton.LEFT:
            return

        _gx = int(math.floor(mouseevent.xdata))
        _gy = int(math.floor(mouseevent.ydata))

        if key == "a":
            if not self.nn_average(_gx, _gy):
                return
        elif key == "d":
            if not self.nn_average(_gx, _gy, avgtype="deepen"):
                return
        elif key == "l":
            # click left mouse button to make a land point
            self.z[_gy, _gx] = self.lnd_val
        elif key == "o":
            # click right mouse button to make a ocean point
            _z = self.z[_gy, _gx]
            if _z < 0:
                return
            self.z[_gy, _gx] = self.min_depth

        self.mesh.set_array(self.z)
        self.fig.canvas.draw()

    def nn_average(self, _gx, _gy, avgtype="normal") -> bool:
        logger.info(f"Averaging ({avgtype}) for point {(_gy,_gx)}")
        ny, nx = self.z.shape
        y1, y2 = max(_gy - 1, 0), min(_gy + 2, ny + 1)
        x1, x2 = max(_gx - 1, 0), min(_gx + 2, nx + 1)
        _z = self.z[y1:y2, x1:x2]
        if avgtype == "deepen":
            z = self.z[_gy, _gx]
            z_ = _z[_z < z]
        else:
            z_ = _z[_z < 0.0]
        logger.info(f"points {z_.shape}, {list(z_)} {(y1,y2,x1,x2)}")
        if len(z_) < 1:
            return False
        self.z[_gy, _gx] = np.mean(z_)
        return True


@app.command("edit_bathy")
def edit_bathy(
    bathy_file: Path = typer.Option(..., file_okay=True, readable=True, dir_okay=False),
    nx: int = typer.Option(...),
    ny: int = typer.Option(...),
):
    """
    Opens up a GUI to click and edit Bathymetry \n
    Keys: \n
        - a: average
        - d: deepen
        - l: land
        - o: ocean
    """
    EditBathy(bathy_file, nx, ny)


app_click = typer.main.get_command(app)
