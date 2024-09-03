#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from tesscube import TESSCube
from tess_asteroid_ml.make_TESS_asteroid_db import *
from tess_asteroid_ml import PACKAGEDIR
from sklearn.preprocessing import QuantileTransformer
from astropy.stats import sigma_clip
from scipy import ndimage

SAT_LEVEL = 1e5


def get_FFI_path(
    file_name, sector: int = 1, camera: int = 1, ccd: int = 1, provider: str = "mast"
):
    if provider == "mast":
        root_path = "https://archive.stsci.edu/missions/tess"
    elif provider == "aws":
        root_path = "s3://stpubdata/tess/public"

    aux = []
    for fn in file_name:
        date_o = fn[4:17]
        yyyy = date_o[:4]
        ddd = date_o[4:7]
        camera = fn.split("-")[2]
        ccd = fn.split("-")[3]
        dir_path = f"ffi/s{sector:04}/{yyyy}/{ddd}/{camera}-{ccd}"
        aux.append(f"{root_path}/{dir_path}/{fn}")

    file_name = aux

    return file_name


def build_data_transformer(
    sector: int = 1,
    camera: int = 1,
    ccd: int = 1,
    plot: bool = True,
    validate: bool = True,
):
    print("Fit data transformer to scale flux values into Quantiles")

    # use tessrip to find th darkest/brightes/mid cadences
    rips = TESSCube(sector=sector, camera=camera, ccd=ccd)
    flux, _ = rips.get_flux(shape=(100, 100))

    nt, nr, nc = flux.shape
    mean_flux = np.mean(flux.reshape((nt, nr * nc)), axis=-1)
    qmask = rips.quality == 0

    pcen = np.percentile(mean_flux[qmask], [0, 1, 50, 99, 100], interpolation="nearest")
    cad_pcen = [np.abs(mean_flux[qmask] - x).argmin() for x in pcen]
    print("\tSelected cadences with percentile (0, 1, 50 99, 100):", cad_pcen)

    # get ffi data from selected frames
    ffi_file = get_FFI_path(
        np.array(rips.ffi_names)[cad_pcen], sector=sector, camera=camera, ccd=ccd
    )
    ffi_header, f2d, col_2d, row_2d, ra_2d, dec_2d = get_data_from_files(
        ffi_file, provider="mast"
    )
    f2d = np.asarray(f2d)

    # fit transformer
    print("\tFitting scaler...")
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="uniform")

    qt.fit(f2d.flatten().reshape(-1, 1))
    flux_quant = qt.transform(f2d.flatten().reshape(-1, 1)).reshape(f2d.shape)
    print(qt)

    # save object
    dir_name = f"{os.path.dirname(PACKAGEDIR)}/data/support/data_transformers"
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    file_out = f"{dir_name}/quantile_transformer_s{sector:04}-{camera}-{ccd}.pkl"
    print(f"\tSaving to {file_out}")
    with open(file_out, "wb") as f:
        pickle.dump(qt, f)

    # validate
    if validate:
        print("\tValidating")
        with open(file_out, "rb") as f:
            qt2 = pickle.load(f)

        flux_quant2 = qt2.transform(f2d.flatten().reshape(-1, 1)).reshape(f2d.shape)

        assert np.isclose(flux_quant, flux_quant2).all()

    # make sat pixel and star mask
    print("Creating pixel mask Sat & Star")
    sat_mask = (f2d >= SAT_LEVEL).any(axis=0)
    sat_mask = ndimage.binary_dilation(sat_mask, iterations=3)

    stack = np.median(f2d, axis=0)
    star_mask = sigma_clip(stack, cenfunc="median", stdfunc="std", sigma=5).mask

    # save mask to npz
    dir_name = f"{os.path.dirname(PACKAGEDIR)}/data/support/pixel_mask"
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    file_out = f"{dir_name}/pixel_masks_s{sector:04}-{camera}-{ccd}.npz"

    print(f"\tSaving to {file_out}")
    np.savez(
        file_out,
        saturation_mask=sat_mask.astype(bool),
        star_mask=star_mask.astype(bool),
    )

    if plot:
        print("Ploting figures...")

        fig, ax = plt.subplots(1, 3, figsize=(20, 4))
        fig.suptitle(f"Sector {sector} Camera {camera} CCD {ccd} Scaling")

        ax[0].set_title("Raw/Linear")
        im = ax[0].pcolormesh(
            col_2d,
            row_2d,
            f2d[1],
            cmap="viridis",
            rasterized=True,
        )
        plt.colorbar(im, ax=ax[0], location="right", shrink=1, label="Flux [-e/s]")

        ax[1].set_title("Quantile Uniform")
        im = ax[1].pcolormesh(
            col_2d,
            row_2d,
            flux_quant[1],
            cmap="viridis",
            rasterized=True,
        )
        plt.colorbar(im, ax=ax[1], location="right", shrink=1, label="Quantile")

        ax[0].set_xlabel("Pixel Column")
        ax[1].set_xlabel("Pixel Column")
        ax[0].set_ylabel("Pixel Row")

        ax[0].set_aspect("equal", "box")
        ax[1].set_aspect("equal", "box")

        ax[2].scatter(
            f2d[1].ravel(),
            flux_quant[1].ravel(),
            marker=".",
            s=0.5,
            alpha=0.5,
            rasterized=True,
        )
        ax[2].set_xscale("log")
        ax[2].set_title("Response Curve")
        ax[2].set_xlabel("Raw Flux [e-/s]")
        ax[2].set_ylabel("Scaled Flux")

        dir_name = f"{os.path.dirname(PACKAGEDIR)}/data/figures"
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        dir_name = f"{dir_name}/tess_ffi_s{sector:04}-{camera}-{ccd}_data_scaling.pdf"
        print(f"\tFigure: {dir_name} ")
        plt.savefig(dir_name, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(1, 2, figsize=(16, 9))

        sat_mask = sat_mask.astype(float)
        sat_mask[sat_mask == 0] = np.nan
        star_mask = star_mask.astype(float)
        star_mask[star_mask == 0] = np.nan

        ax[0].set_title("Saturated Pixel Mask")
        ax[0].pcolormesh(col_2d, row_2d, flux_quant[1], rasterized=True)
        ax[0].pcolormesh(
            col_2d,
            row_2d,
            sat_mask,
            alpha=0.5,
            cmap="Reds",
            rasterized=True,
            vmin=0,
            vmax=1,
        )
        ax[0].set_aspect("equal", "box")

        ax[1].set_title("Star Pixel Mask")
        ax[1].pcolormesh(col_2d, row_2d, flux_quant[1], rasterized=True)
        ax[1].pcolormesh(
            col_2d,
            row_2d,
            star_mask,
            alpha=0.5,
            cmap="Reds",
            rasterized=True,
            vmin=0,
            vmax=1,
        )
        ax[1].set_aspect("equal", "box")

        dir_name = f"{os.path.dirname(PACKAGEDIR)}/data/figures"
        dir_name = f"{dir_name}/tess_ffi_s{sector:04}-{camera}-{ccd}_pixel_mask.pdf"
        print(f"\tFigure: {dir_name} ")
        plt.savefig(dir_name, bbox_inches="tight")
        plt.close()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates a dataset from TESS FFIs. "
        "Makes 64x64 pixel cuts, uses JPL to create a asteroid mask."
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
        "--plot",
        dest="plot",
        action="store_true",
        default=False,
        help="Plot FFI data transformation.",
    )
    parser.add_argument(
        "--validate",
        dest="validate",
        action="store_true",
        default=False,
        help="Validate pickle object.",
    )
    args = parser.parse_args()
    build_data_transformer(
        sector=args.sector,
        camera=args.camera,
        ccd=args.ccd,
        plot=args.plot,
        validate=args.validate,
    )
