import os
import argparse

# import nest_asyncio
import numpy as np
import pandas as pd
import lightkurve as lk
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io import fits
from tqdm import tqdm
from astrocut import CutoutFactory

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
            # print(f"Creating FFI cut: {output}")
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
):
    """"""

    track_file_root = f"{os.path.dirname(PACKAGEDIR)}/data/jpl/tracks/sector{sector:04}"
    if not os.path.isdir(track_file_root):
        raise ValueError(f"No asteroid db for sector {sector}")

    sb_ephems = {}
    for k, row in tqdm(df.iterrows(), total=len(df), desc="Asteroid list", leave=False):
        # read file with asteroid track from disk if exists
        track_file = (
            f"{track_file_root}/"
            f"tess-ffi_s{sector:04}-0-0_{row['id'].replace(' ', '-')}_hires.feather"
        )
        if os.path.isfile(track_file):
            ephems_aux = pd.read_feather(track_file).query(
                f"camera == {camera} and ccd == {ccd}"
            )
            if low_res:
                step = len(ephems_aux) // 27 - 1 if len(ephems_aux) > 100 else 4
                ephems_aux = ephems_aux[::step]
            sb_ephems[k] = ephems_aux
    return sb_ephems


def create_cutout_data(
    sector: int = 1,
    camera: int = 1,
    ccd: int = 1,
    cutout_size: int = 50,
    maglim: float = 22,
    provider: str = "mast",
):
    # get FFI data to build ra/dec coordinates for later query
    ffi_file = get_FFI_name(sector=sector, camera=camera, ccd=ccd, provider=provider)
    ffi_header, f2d, col_2d, row_2d, ra_2d, dec_2d = get_data_from_files(
        ffi_file, provider=provider
    )
    ffi_header = ffi_header[0]
    f2d = f2d[0]
    col_2d = col_2d[0]
    row_2d = row_2d[0]
    ra_2d = ra_2d[0]
    dec_2d = dec_2d[0]
    jpl_df = get_asteroid_table(
        SkyCoord(ra_2d.min() * u.deg, dec_2d.min() * u.deg, frame='icrs'),
        SkyCoord(ra_2d.max() * u.deg, dec_2d.max() * u.deg, frame='icrs'),
        sector=sector,
        camera=camera,
        ccd=ccd,
    )
    if maglim <= 30:
        asteroid_df = jpl_df.query("V_mag <= 18")

    sb_ephems_lowres = read_asteroid_db(
        asteroid_df, low_res=True, sector=sector, camera=camera, ccd=ccd
    )

    xcen, ycen = get_cutout_centers(sampling="tiled", overlap=5, size=cutout_size)

    for i in ycen[1:, 0]:
        sb_ephems_highres = {}
        for j in xcen[0, 1:]:
            cut_dict = {}
            cut_dict[f"c{col_2d[i, j]:04}_r{row_2d[i,j]:04}"] = SkyCoord(
                ra_2d[i, j] * u.deg, dec_2d[i, j] * u.deg, frame="icrs"
            )
            tpf_names = get_cutouts(
                cut_dict,
                sector=sector,
                cam_ccd=f"{camera}-{ccd}",
                cutout_size=cutout_size,
            )[0]
            fficut_aster = AsteroidTESScut(lk.read(tpf_names))
            fficut_aster.ffi_exp_time = ffi_header["EXPOSURE"] * 24 * 3600
            fficut_aster.get_CBVs(align=False, interpolate=True)

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
                    print(
                        f"{k} Asteroid in cutout: ", jpl_sb_bright.loc[k, "query_name"]
                    )
                    # predict full res track
                    if k not in sb_ephems_highres.keys():
                        sb_ephems_lowres = read_asteroid_db(
                            jpl_sb_bright.loc[k],
                            low_res=False,
                            sector=sector,
                            camera=camera,
                            ccd=ccd,
                        )
                    source_rad = 2e2 / (sb_ephems_highres[k].vmag.mean()) ** 1.8
                    fficut_aster.get_asteroid_mask(
                        sb_ephems_highres[k],
                        name=jpl_sb_bright.loc[k, "Object name"],
                        mask_type="circular",
                        mask_radius=source_rad,
                    )
            fficut_aster.save_data()
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
        "--cutout-size",
        dest="cutout_size",
        type=int,
        default=None,
        help="Cutout size in pixels",
    )
    args = parser.parse_args()
    create_FFI_asteroid_database(
        sector=args.sector,
        camera=args.camera,
        ccd=args.ccd,
        plot=False,
    )
