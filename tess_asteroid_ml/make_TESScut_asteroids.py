import os, sys
import re
import argparse
import pickle
import tarfile
import tempfile
from glob import glob

import numpy as np
import pandas as pd
import lightkurve as lk
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from tqdm import tqdm
from astrocut import CutoutFactory
from astropy.stats import sigma_clip

from tess_asteroid_ml import *
from tess_asteroid_ml.make_TESS_asteroid_db import *
from tess_asteroid_ml.utils import in_cutout


def get_cutouts(
    target_dict: dict, sector: int = 1, cam_ccd: str = "1-1", cutout_size=50
):
    """
    Downloads a TESS cut using AstroCut
    """
    # We need nest_asyncio for AWS access within Jupyter
    # nest_asyncio.apply()
    # Create a "cutout factory" to generate cutouts
    factory = CutoutFactory()
    cube_file = (
        f"s3://stpubdata/tess/public/mast/tess-s{sector:04d}-{cam_ccd}-cube.fits"
    )
    tpf_path = f"{os.path.dirname(PACKAGEDIR)}/data/tesscuts/sector{sector:04}"
    if not os.path.isdir(tpf_path):
        os.makedirs(tpf_path)
    tpf_names = []
    for k, v in target_dict.items():
        output = (
            f"{tpf_path}/"
            f"TESScut_s{sector:04}-{cam_ccd}_{k}_{cutout_size}x{cutout_size}pix.fits"
        )
        if not os.path.isfile(output):
            print(f"Creating FFI cut: {output}")
            factory.cube_cut(
                cube_file=cube_file,
                coordinates=v,
                cutout_size=cutout_size,
                target_pixel_file=output,
                threads="auto",
            )
        tpf_names.append(output)

    return tpf_names


def get_cutout_centers(
    ncol: int = 2048,
    nrow: int = 2048,
    sampling: str = "tiled",
    overlap: int = 0,
    size: int = 50,
    ncuts: int = 20,
):
    """
    Get the i,j of the centers for a sample of cutouts given a cutout size and a
    sampling strategy.
    """
    if sampling == "tiled":
        dx = 2
        mid = size // 2
        xcen = np.arange(dx + mid, ncol - mid, size - overlap)
        ycen = np.arange(dx + mid, nrow - mid, size - overlap)
    elif sampling == "random":
        xcen, ycen = np.random.randint(0 + dx, ncol - dx, (2, ncuts))
    else:
        raise NotImplementedError
    # xcen = xcen[xcen < ncol]
    # ycen = ycen[ycen < nrow]
    xcen, ycen = np.meshgrid(xcen, ycen)

    return xcen, ycen


def read_asteroid_db(
    df,
    low_res: bool = False,
    sector: int = 1,
    camera: int = 0,
    ccd: int = 0,
    quiet: bool = False,
):
    """"""

    tarf = f"{os.path.dirname(PACKAGEDIR)}/data/jpl/tracks/sector{sector:04}.tar"
    if not os.path.isfile(tarf):
        raise ValueError(f"No asteroid db for sector {sector}")

    sb_ephems = {}
    with tempfile.TemporaryDirectory(prefix="temp_fits") as tmpdir:
        tardb = tarfile.open(tarf, mode="r")
        tardb_names = tardb.getnames()
        for k, row in tqdm(
            df.iterrows(), total=len(df), desc="Asteroid list", leave=False, disable=quiet
        ):
            # read file with asteroid track from disk if exists
            track_file = (
                f"sector{sector:04}/"
                f"tess-ffi_s{sector:04}-0-0_{row['id'].replace(' ', '-')}_hires.feather"
            )
            if track_file in tardb_names:
                tardb.extract(track_file, tmpdir)
                ephems_aux = pd.read_feather(f"{tmpdir}/{track_file}")
                if camera != 0:
                    ephems_aux = ephems_aux.query(f"camera == {camera}")
                if ccd != 0:
                    ephems_aux = ephems_aux.query(f"ccd == {ccd}")
                if low_res:
                    step = len(ephems_aux) // 27 - 1 if len(ephems_aux) > 100 else 4
                    ephems_aux = ephems_aux[::step]
                sb_ephems[k] = ephems_aux
    return sb_ephems


def get_ffi_cutouts(
    coords: dict = {},
    sampling: str = "sparse",
    download: bool = False,
    sector: int = 1,
    cam_ccd: str = "1-1",
    cutout_size: int = 64,
):
    tpf_names_list = []

    nrows = np.unique([int(x[-4:]) for x in coords.keys()])
    for k, row in enumerate(nrows):
        if sampling == "sparse" and k + 1 not in [3, 5, 8, 11, 15, 19, 25]:
            # will only create/save data for a few rows in the FFI tile
            tpf_names_list.append([])
            continue
        row_dict = {k: v for k, v in coords.items() if f"r{row:04}" in k}
        if download:
            print(
                "WARNING: this will query Astrocut/MAST to create and download "
                f"{len(row_dict)} cutouts. Only run this onece, then use saved files."
            )
            tpf_names = get_cutouts(
                row_dict,
                sector=sector,
                cam_ccd=cam_ccd,
                cutout_size=cutout_size,
            )
        else:
            tpf_names = glob(
                f"{os.path.dirname(PACKAGEDIR)}/data/tesscuts/sector{sector:04}"
                f"/TESScut_s{sector:04}-{cam_ccd}_c*_r{row:04}_{cutout_size}x{cutout_size}pix.fits"
            )

        tpf_names_list.append(tpf_names)
    return tpf_names_list


def make_asteroid_cut_data(
    sector: int = 1,
    camera: int = 1,
    ccd: int = 1,
    sampling: str = "sparse",
    fit_bkg: bool = False,
    limiting_mag: float = 22,
    cutout_size: int = 64,
    verbose: bool = False,
    plot: bool = False,
    download=False,
):
    provider = "mast"

    # get FFI image
    ffi_file = get_FFI_name(
        sector=sector, camera=camera, ccd=ccd, provider=provider, correct=False
    )
    if verbose:
        print(f"Working on FFI sector {sector} camera {camera} ccd {ccd}")
        print(f"FFI file is {ffi_file[0]}")

    # get FFI data to build ra/dec coordinates for later query
    ffi_header, f2d, col_2d, row_2d, ra_2d, dec_2d = get_data_from_files(
        ffi_file, provider=provider
    )
    ffi_header = ffi_header[0]
    f2d = f2d[0]
    col_2d = col_2d
    row_2d = row_2d
    ra_2d = ra_2d[0]
    dec_2d = dec_2d[0]

    obs_time = Time([ffi_header["DATE-OBS"], ffi_header["DATE-END"]], format="isot")

    # get asteroid catalog
    jpl_df = get_asteroid_table(
        SkyCoord(ra_2d.min() * u.deg, dec_2d.min() * u.deg, frame='icrs'),
        SkyCoord(ra_2d.max() * u.deg, dec_2d.max() * u.deg, frame='icrs'),
        sector=sector,
        camera=camera,
        ccd=0,
        date_obs=obs_time.mean().jd,
    )
    if limiting_mag <= 30:
        asteroid_df = jpl_df.query(f"V_mag <= {limiting_mag}")
    if verbose:
        print(f"Asteroid table has {len(asteroid_df)} items with V < {limiting_mag}")

    # get asteroid tracks in low res
    sb_ephems_lowres = read_asteroid_db(
        asteroid_df, low_res=True, sector=sector, camera=camera, ccd=ccd
    )
    if verbose:
        print(f"Asteroid track DB has {len(sb_ephems_lowres)} available")

    # plot FFI image with asteroid tracks if asked
    if plot:
        if verbose:
            print("Plotting FFI image to file")
        vlo, lo, mid, hi, vhi = np.nanpercentile(f2d, [0.2, 3, 50, 95, 99])
        cnorm = colors.LogNorm(vmin=lo, vmax=vhi)

        plt.figure(figsize=(7, 7))
        plt.pcolormesh(col_2d, row_2d, f2d, norm=cnorm, cmap="Greys_r", rasterized=True)

        counter = 0
        for k, val in sb_ephems_lowres.items():
            if len(val) == 0:
                continue
            val = val.query(f"camera == {camera} and ccd == {ccd}")
            if len(val) > 0:
                counter += 1
            plt.plot(val.column, val.row, "-", ms=0.8, lw=0.5, rasterized=True)
            # if k == 1: break
        plt.title(
            f"Asteroid tracks in Sector {sector} Camera {camera} CCD {ccd} \n "
            f"V < {limiting_mag} N = {counter}"
        )

        plt.xlim(col_2d.min() - 10, col_2d.max() + 10)
        plt.ylim(row_2d.min() - 10, row_2d.max() + 10)
        # plt.gca().set_aspect('equal')
        plt.xlabel("Pixel Column")
        plt.ylabel("Pixel Row")
        # plt.show()
        dir_name = f"{os.path.dirname(PACKAGEDIR)}/data/figures"
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        dir_name = (
            f"{dir_name}/tess_ffi_s{sector:04}-{camera}-{ccd}_asteroid_tracks.pdf"
        )
        plt.savefig(dir_name, bbox_inches="tight")
        if verbose:
            print(f"Figure saved to {dir_name}")
        plt.close()

    # get cutout centers given size and overlap
    xcen, ycen = get_cutout_centers(sampling="tiled", overlap=5, size=cutout_size)
    # create a dict with cutout centers in radec
    cut_dict = {}
    for i in ycen[1:, 0]:
        for j in xcen[0, 1:]:
            cut_dict[f"c{col_2d[i, j]:04}_r{row_2d[i,j]:04}"] = SkyCoord(
                ra_2d[i, j] * u.deg, dec_2d[i, j] * u.deg, frame="icrs"
            )

    tpf_names_list = get_ffi_cutouts(
        coords=cut_dict,
        sampling=sampling,
        download=download,
        sector=sector,
        cam_ccd=f"{camera}-{ccd}",
        cutout_size=cutout_size,
    )
    if np.max([len(x) for x in tpf_names_list]) == 0:
        print("WARNING: No cutout TPFs available in disk. Please run again with "
              "`--download` flag to get the data")

    # iterate over cutouts row in the grid
    for nn, tpf_names in enumerate(tpf_names_list):
        if len(tpf_names) == 0:
            continue
        nrow = np.unique(
            [int(re.findall(r"\_c(\d+)\_r(\d+)\_", string=x)[0][1]) for x in tpf_names]
        )[0]
        if verbose:
            print(
                f"Working with FFI row {nn + 1}/{len(tpf_names_list)} "
                f"and {len(tpf_names)} cuts..."
            )
        # iterate cutouts in a single row
        F, X, Y, L, NAMES = [], [], [], [], []
        sb_ephems_highres = {}
        for q, ff in tqdm(
            enumerate(tpf_names),
            total=len(tpf_names),
            desc=f"TESS cut files row {nrow + 1}",
        ):
            fficut_aster = AsteroidTESScut(lk.read(ff, quality_bitmask=None))
            fficut_aster.ffi_exp_time = (
                (ffi_header["TSTOP"] - ffi_header["TSTART"]) * 24 * 3600
            )
            fficut_aster.get_CBVs(align=False, interpolate=True)
            fficut_aster.get_quaternions_and_angles()
            if fit_bkg:
                fficut_aster.fit_background(polyorder=3, positive_flux=True)

            for k, val in sb_ephems_lowres.items():
                if len(val) == 0:
                    continue
                # check if asteroid track passes over the TESScut
                is_in = in_cutout(
                    fficut_aster.column,
                    fficut_aster.row,
                    sb_ephems_lowres[k].column.values,
                    sb_ephems_lowres[k].row.values,
                )
                if is_in:
                    if k not in sb_ephems_highres.keys():
                        sb_ephems_highres.update(
                            read_asteroid_db(
                                asteroid_df.loc[[k]],
                                low_res=False,
                                sector=sector,
                                camera=camera,
                                ccd=ccd,
                                quiet=True,
                            )
                        )
                    source_rad = 3.2e2 / (sb_ephems_highres[k].vmag.mean()) ** 1.8
                    fficut_aster.get_asteroid_mask(
                        sb_ephems_highres[k],
                        name=asteroid_df.loc[k, ["Object name", "V_mag"]],
                        mask_type="circular",
                        mask_radius=source_rad,
                        mask_num_type="dec",
                    )
            F.append(fficut_aster.flux_2d)
            X.append(fficut_aster.column_2d[0, 0])
            Y.append(fficut_aster.row_2d[0, 0])
            L.append(fficut_aster.asteroid_mask_2d)
            if hasattr(fficut_aster, "asteroid_names"):
                NAMES.append(
                    pd.DataFrame.from_dict(fficut_aster.asteroid_names, orient="index")
                )
            else:
                NAMES.append(pd.DataFrame([]))

            if fficut_aster.asteroid_mask_2d is None:
                break

        # break data into orbits to ensure continuous obs
        fficut_aster.find_orbit_breaks()
        F, X, Y, L = np.array(F), np.array(X), np.array(Y), np.array(L)
        quat_nonnan_mask = np.isfinite(fficut_aster.quaternions[:, 0])
        F = F[:, quat_nonnan_mask]
        L = L[:, quat_nonnan_mask]
        TIME = fficut_aster.time[quat_nonnan_mask]
        CBV = fficut_aster.cbvs[quat_nonnan_mask]
        QUAT = fficut_aster.quaternions[quat_nonnan_mask]
        E_ANG = fficut_aster.earth_angle[quat_nonnan_mask]
        M_ANG = fficut_aster.moon_angle[quat_nonnan_mask]

        dts = np.diff(TIME)
        breaks = np.where(dts >= 0.2)[0] + 1

        if len(breaks) > 0:
            F = np.array_split(F, breaks, axis=1)
            L = np.array_split(L, breaks, axis=1)
            TIME = np.array_split(TIME, breaks, axis=0)
            CBV = np.array_split(CBV, breaks, axis=0)
            QUAT = np.array_split(QUAT, breaks, axis=0)
            E_ANG = np.array_split(E_ANG, breaks, axis=0)
            M_ANG = np.array_split(M_ANG, breaks, axis=0)
        else:
            F = [F]
            L = [L]
            TIME = [fficut_aster.time]
            CBV = [fficut_aster.cbvs]
            QUAT = [fficut_aster.quaternions]
            E_ANG = [fficut_aster.earth_angle]
            M_ANG = [fficut_aster.moon_angle]

        # save data to disk
        out_path = f"{os.path.dirname(PACKAGEDIR)}/data/asteroidcuts/sector{sector:04}"
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        # save asteroid catalog per cutout
        out_file = (
            f"{out_path}/tess-asteroid-cuts_{cutout_size}x{cutout_size}"
            f"_s{sector:04}-{camera}-{ccd}_V{limiting_mag}_{nrow:02}.pkl"
        )
        with open(out_file, "wb") as outp:
            pickle.dump(NAMES, outp, -1)

        # save flux, mask and array data as npz files per row of cutouts per orbit
        for bk in range(len(F)):
            if len(TIME[bk]) < cutout_size + 2:
                continue
            out_file = (
                f"{out_path}/tess-asteroid-cuts_{cutout_size}x{cutout_size}"
                f"_s{sector:04}-{camera}-{ccd}_V{limiting_mag}_orb{bk+1}_bkg{str(fit_bkg)[0]}_{nrow:02}.npz"
            )
            np.savez(
                out_file,
                flux=F[bk],
                column=X,
                row=Y,
                mask=L[bk].astype(np.int16),
                time=TIME[bk],
                cbv=CBV[bk],
                quat=QUAT[bk],
                earth_angles=E_ANG[bk],
                moon_angles=M_ANG[bk],
            )

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
        "--cutout-size",
        dest="cutout_size",
        type=int,
        default=64,
        help="Cutout size in pixels",
    )
    parser.add_argument(
        "--lim-mag",
        dest="lim_mag",
        type=float,
        default=22.0,
        help="Limiting magnitude in V band.",
    )
    parser.add_argument(
        "--sampling",
        dest="sampling",
        type=str,
        default="sparse",
        help=("Select a `dense` grid that covers corner to corner of the FFI or a "
             "`sparse` that uses only 7 fixed rows from the grid."),
    )
    parser.add_argument(
        "--fit-bkg",
        dest="fit_bkg",
        action="store_true",
        default=False,
        help="Fit and substract background (flag).",
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        default=False,
        help="Plot FFI + asteroid tracks  (flag).",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Verbose (flag).",
    )
    parser.add_argument(
        "--download",
        dest="download",
        action="store_true",
        default=False,
        help="Donwload cutouts from from AWS with Astrocut (flag).",
    )
    args = parser.parse_args()
    make_asteroid_cut_data(
        sector=args.sector,
        camera=args.camera,
        ccd=args.ccd,
        sampling=args.sampling,
        fit_bkg=args.fit_bkg,
        limiting_mag=args.lim_mag,
        cutout_size=args.cutout_size,
        verbose=args.verbose,
        plot=args.plot,
        download=args.download,
    )
