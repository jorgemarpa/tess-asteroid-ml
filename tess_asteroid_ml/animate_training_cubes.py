import argparse
import os
import socket
from glob import glob

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from tqdm.notebook import tqdm

from tess_asteroid_ml import PACKAGEDIR

if "adapt" in socket.gethostname():
    DATAPATH = "/explore/nobackup/projects/asteroid/data/asteroidcuts"
elif "fortuna" in socket.gethostname():
    DATAPATH = "/Users/jimartin/Work/TESS/tess-asteroid-ml/data/asteroidcuts"
else:
    DATAPATH = "/Users/jimartin/Work/TESS/tess-asteroid-ml/data/asteroidcuts"


# naive rolling median window in time
# this might not be efficient
def build_static(cube, window=50):
    nt, nx, ny = cube.shape
    df = pd.DataFrame(cube.reshape(nt, nx * ny))
    rmed = (
        df.rolling(window, min_periods=10, center=True, closed="both")
        .median()
        .values.reshape(nt, nx, ny)
    )
    # silly solution of repeating the first 50 frames to match the shape
    # this needs to be better, maybe interpolation?
    # med = np.median(cube[:window], axis=0)
    # rmed[:window-1] = np.repeat(med.reshape(-1, nx, ny), window-1, axis=0)
    return rmed


def plot_img_aperture(
    col,
    row,
    img,
    aperture_mask,
    ax=None,
    cbar=False,
    vmin=0,
    vmax=1,
    cadno=1,
    time=2457000,
):
    if ax is None:
        fig, ax = plt.subplots()
        fig.suptitle("Asteroid Mask")
    im = ax.pcolormesh(
        col,
        row,
        img,
        cmap="Greys_r",
        vmin=vmin,
        vmax=vmax,
        # norm=cnorm,
        rasterized=True,
    )
    ax.set_aspect("equal", "box")
    ax.set_title(f"CAD {cadno} | BTJD {time - 2457000:.4f}")
    if cbar:
        plt.colorbar(im, location="right", shrink=0.8, label="Quantile Scaled Flux")

    for i, pi in enumerate(row[:, 0]):
        for j, pj in enumerate(col[0, :]):
            if aperture_mask[i, j]:
                # print("here")
                rect = patches.Rectangle(
                    xy=(pj - 0.5, pi - 0.5),
                    width=1,
                    height=1,
                    color="tab:red",
                    fill=False,
                    alpha=0.4,
                )
                ax.add_patch(rect)

    ax.set_xlabel("Pixel Column")
    ax.set_ylabel("Pixel Row")

    return ax


def animate_image(
    col,
    row,
    cube,
    aperture_mask,
    interval=200,
    repeat_delay=1000,
    sector=1,
    camera=1,
    ccd=1,
    cadenceno=None,
    time=None,
):
    vlo, lo, mid, hi, vhi = np.nanpercentile(cube, [1, 5, 50, 95, 99.8])
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(f"Asteroid Masks in Sector {sector} Camera {camera} CCD {ccd}")

    ax = plot_img_aperture(
        col,
        row,
        cube[0],
        aperture_mask[0],
        ax=ax,
        cbar=True,
        vmin=lo,
        vmax=vhi,
        cadno=cadenceno[0],
        time=time[0],
    )

    def animate(nt):
        ax.clear()
        _ = plot_img_aperture(
            col,
            row,
            cube[nt],
            aperture_mask[nt],
            ax=ax,
            cbar=False,
            vmin=lo,
            vmax=vhi,
            cadno=cadenceno[nt],
            time=time[nt],
        )

        return ()

    plt.close(ax.figure)

    # Create the animation
    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(cube),
        interval=interval,
        blit=True,
        repeat_delay=repeat_delay,
        repeat=True,
    )

    return ani


def animate_cube(sector=1, camera=1, ccd=1, orb=1, ncubes=5):
    forb = [
        sorted(
            glob(
                f"{DATAPATH}/sector{sector:04}/tess-asteroid-cuts_*_s{sector:04}-{camera}-{ccd}*orb{k}*.npz"
            )
        )
        for k in range(orb, orb + 1)
    ]
    forb = np.asarray([x for x in forb if x != []]).T

    print(f"Total files in directory {forb.shape[0]} with {forb.shape[1]} orbits")

    rnd_fc = np.random.randint(0, 34, size=ncubes)

    dir_name = (
        f"{os.path.dirname(PACKAGEDIR)}/data/figures/training_cube/sector{sector:04}"
    )
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    for nf in tqdm(rnd_fc, total=len(rnd_fc)):
        mask = []
        flux = []
        col = []
        row = []
        time = []
        cadno = []

        for f in forb[nf]:
            data = np.load(f)

            mask.append(data["mask"].transpose(1, 0, 2, 3))
            flux.append(data["flux"].transpose(1, 0, 2, 3))
            col = data["column"]
            row = data["row"]
            time.extend(data["time"])
            cadno.extend(data["cadenceno"])

        mask = np.vstack(mask).transpose(1, 0, 2, 3)
        flux = np.vstack(flux).transpose(1, 0, 2, 3)

        time = np.array(time)
        cadno = np.array(cadno)
        rnd_nc = np.random.randint(0, 34, size=ncubes)

        for nc in rnd_nc:
            cube_flux = flux[nc]
            cube_mask = mask[nc]

            cube_row2d, cube_col2d = np.mgrid[
                row[nc] : row[nc] + 64, col[nc] : col[nc] + 64
            ]

            # diff = cube_flux - np.nanmedian(cube_flux, axis=0)
            diff = cube_flux - build_static(cube_flux, window=99)

            # save animation to movie file
            animate_image(
                cube_col2d,
                cube_row2d,
                diff,
                cube_mask,
                interval=75,
                cadenceno=cadno,
                time=time,
            ).save(
                f"{dir_name}/cutout_training_{camera}-{ccd}_animation_{nf}-{nc}-orb{orb}.gif",
                writer="pillow",
            )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates a dataset from TESS FFIs. "
        "Makes 50x50 pixel cuts, uses JPL to create a asteroid mask."
    )
    parser.add_argument(
        "--sector",
        dest="sector",
        type=int,
        default=None,
        help="TESS sector number.",
    )
    parser.add_argument(
        "--camera",
        dest="camera",
        type=int,
        default=None,
        help="TESS camera number.",
    )
    parser.add_argument(
        "--ccd",
        dest="ccd",
        type=int,
        default=None,
        help="TESS CCD number",
    )
    parser.add_argument(
        "--orbit",
        dest="orbit",
        type=int,
        default=1,
        help="TESS orbit",
    )
    args = parser.parse_args()
    animate_cube(
        sector=args.sector,
        camera=args.camera,
        ccd=args.ccd,
        orb=args.orbit,
    )
