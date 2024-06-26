import socket
import argparse
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

LOCAL = "/Users/jorgemarpa/Work/BAERI/ADAP/tess-asteroid-ml/"
ADAPT = "/explore/nobackup/projects/asteroid/data/asteroidcuts"
JIMARTIN = "/home/jimartin/tess-asteroid-ml/"

if "Jorge" in socket.gethostname():
    DATAPATH = "/Users/jorgemarpa/Work/BAERI/ADAP/tess-asteroid-ml/data/asteroidcuts"
    FIGUREPATH = "/Users/jorgemarpa/Work/BAERI/ADAP/tess-asteroid-ml/data/figures"
elif "adapt" in socket.gethostname():
    DATAPATH = "/explore/nobackup/projects/asteroid/data/asteroidcuts"
    FIGUREPATH = "/home/jimartin/tess-asteroid-ml/data/figures"


def flatten_list(nested_list):
    """
    Flattens a nested list using list comprehension.

    Args:

        nested_list (list): The nested list to be flattened.

    Returns:
        list: A flattened version of the input list.
    """
    return [item for sublist in nested_list for item in sublist]



def plot_mask_cube(sector: int=1, camera: int=1, ccd: int=1):

    forb = [
        sorted(
            glob(
                f"{DATAPATH}/sector{sector:04}/tess-asteroid-cuts_*_s{sector:04}-{camera}-{ccd}*orb{k}*.npz"
            )
        )
        for k in range(1, 5)
    ]
    forb = np.asarray([x for x in forb if x != []]).T

    cube_mask_sum = []
    cube_row = []
    cube_col = []

    for fs in forb:
        mask_aux = []
        for f in fs:
            data = np.load(f)
            mask_aux.append(data["mask"].astype(bool).astype(int).sum(axis=1))

        cube_col.extend(data["column"])
        cube_row.extend(data["row"])
        cube_mask_sum.append(np.array(mask_aux).sum(axis=0))

    cube_mask_sum = flatten_list(cube_mask_sum)

    cube_row2d = []
    cube_col2d = []

    for r, c in zip(cube_row, cube_col):
        _row2d, _col2d = np.mgrid[r : r + 64, c : c + 64]
        cube_row2d.append(_row2d)
        cube_col2d.append(_col2d)

    cube_mask_sum = np.array(cube_mask_sum)
    cube_row2d = np.array(cube_row2d)
    cube_col2d = np.array(cube_col2d)

    print(cube_mask_sum.shape, cube_row2d.shape, cube_col2d.shape)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    fig.suptitle(f"Sector {sector} Camera {camera} CDD {ccd}", y=0.92)

    for k in range(len(cube_mask_sum)):
        ax.set_title(f"vmax = {4}")
        ax.pcolormesh(
            cube_col2d[k],
            cube_row2d[k],
            cube_mask_sum[k],
            cmap="viridis",
            vmin=0,
            vmax=2,
        )

    # ax.set_ylim(0,1000)
    ax.set_aspect("equal", "box")
    ax.invert_yaxis()
    ax.set_xlabel("Pixel Column")
    ax.set_ylabel("Pixel Row")

    ax.vlines(np.unique(cube_col), 0, 2048, colors="tab:red", alpha=0.7, lw=0.5)
    ax.hlines(np.unique(cube_row), 44, 2092, colors="tab:red", alpha=0.7, lw=0.5)

    print(f"{FIGUREPATH}/asteroid_mask_s{sector:04}-{camera}-{ccd}.png")
    plt.savefig(
        f"{FIGUREPATH}/asteroid_mask_s{sector:04}-{camera}-{ccd}.png",
        bbox_inches="tight",
    )
    # plt.show()

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
    args = parser.parse_args()
    plot_mask_cube(
        sector=args.sector,
        camera=args.camera,
        ccd=args.ccd,
    )
