import os
import argparse
import fitsio
import nest_asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import patches
import lightkurve as lk
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm
from astrocut import CutoutFactory

from sbident import SBIdent

from tess_ephem import ephem, TessEphem
from tess_asteroid_ml import *
from tess_asteroid_ml.utils import in_cutout


def _load_ffi_image(
    telescope,
    fname,
    extension,
    cutout_size=None,
    cutout_origin=[0, 0],
    return_coords=False,
):
    """
    Use fitsio to load an image and return positions and flux values.
    It can do a smal cutout of size `cutout_size` with a defined origin.

    Parameters
    ----------
    telescope: str
        String for the telescope
    fname: str
        Path to the filename
    extension: int
        Extension to cut out of the image
    cutout_size : int
        Size of the cutout in pixels (e.g. 200)
    cutout_origin : tuple of ints
        Origin coordinates in pixels from where the cutout stars. Pattern is
        [row, column].
    return_coords : bool
        Return or not pixel coordinates.

    Return
    ------
    f: np.ndarray
        Array of flux values read from the file. Shape is [row, column].
    col_2d: np.ndarray
        Array of column values read from the file. Shape is [row, column]. Optional.
    row_2d: np.ndarray
        Array of row values read from the file. Shape is [row, column]. Optional.
    """
    f = fitsio.FITS(fname)[extension]
    if telescope.lower() == "kepler":
        # CCD overscan for Kepler
        r_min = 20
        r_max = 1044
        c_min = 12
        c_max = 1112
    elif telescope.lower() == "tess":
        # CCD overscan for TESS
        r_min = 0
        r_max = 2048
        c_min = 44
        c_max = 2092
    else:
        raise TypeError("File is not from Kepler or TESS mission")

    # If the image dimension is not the FFI shape, we change the r_max and c_max
    dims = f.get_dims()
    if dims == [r_max, c_max]:
        r_max, c_max = np.asarray(dims)
    r_min += cutout_origin[0]
    c_min += cutout_origin[1]
    if (r_min > r_max) | (c_min > c_max):
        raise ValueError("`cutout_origin` must be within the image.")
    if cutout_size is not None:
        r_max = np.min([r_min + cutout_size, r_max])
        c_max = np.min([c_min + cutout_size, c_max])
    if return_coords:
        row_2d, col_2d = np.mgrid[r_min:r_max, c_min:c_max]
        return col_2d, row_2d, f[r_min:r_max, c_min:c_max]
    return f[r_min:r_max, c_min:c_max]


def get_cutouts(
    target_dict: dict, sector: int = 1, cam_ccd: str = "1-1", cutout_size=50
):
    """
    Downloads a TESS cut using AstroCut
    """
    # We need nest_asyncio for AWS access within Jupyter
    nest_asyncio.apply()
    # Create a "cutout factory" to generate cutouts
    factory = CutoutFactory()
    cube_file = (
        f"s3://stpubdata/tess/public/mast/tess-s{sector:04d}-{camera_ccd}-cube.fits"
    )
    for k, v in target_dict.items():
        factory.cube_cut(
            cube_file=cube_file,
            coordinates=v,
            cutout_size=cutout_size,
            target_pixel_file=f"./data/TESScut_s{sector:04}-{cam_ccd}_{k}_{cutout_size}x{cutout_size}pix.fits",
        )


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
        xcen = np.arange(dx, ncol - dx, size - overlap)
        ycen = np.arange(dx, nrow - dx, size - overlap)
    elif sampling == "random":
        xcen, ycen = np.random.randint(0 + dx, ncol - dx, (2, ncuts))
    else:
        raise NotImplementedError
    xcen, ycen = np.meshgrid(xcen, ycen)
    return xcen, ycen


def query_jpl_sbi(
    edge1: SkyCoord, edge2: SkyCoord, obstime: float = 2459490, maglim: float = 24
):
    print("Requesting JPL Smal-bodies API")

    # get state of TESS (-95) from Horizons at our observation time
    # and convert it from [AU, AU/day] to [km, km/s]
    # location 500 is geocentric, minor planet center.
    # 500@-95 means Geocentric location to TESS

    au = (1 * u.au).to(u.km).value  # 1AU in km
    tess = Horizons(id='-95', location='500', epochs=obstime, id_type=None).vectors(
        refplane='earth'
    )  # state vector
    tess_km = (
        tess[['x', 'y', 'z', 'vx', 'vy', 'vz']].to_pandas().to_numpy() * au
    )  # convert to km/d
    tess_km[:, 3:] = tess_km[:, 3:] / 86400  # convert to km/s
    tess_km = tess_km[0]  # take the first row

    # form the xobs dictionary that is the input for SBIdent location argument
    xobs = ','.join([np.format_float_scientific(s, precision=5) for s in tess_km])
    xobs_location = {'xobs': xobs}

    sbid3 = SBIdent(
        location=xobs_location,
        obstime=obstime,
        fov=[edge1, edge2],
        maglim=maglim,
        precision="high",
        request=True,
    )

    jpl_sb = sbid3.results.to_pandas()

    # parse columns
    jpl_sb["Astrometric Dec (dd mm\'ss\")"] = [
        x.replace(" ", ":").replace("\'", ":").replace('"', "")
        for x in jpl_sb["Astrometric Dec (dd mm\'ss\")"]
    ]
    coord = SkyCoord(
        jpl_sb[["Astrometric RA (hh:mm:ss)", "Astrometric Dec (dd mm\'ss\")"]].values,
        frame='icrs',
        unit=(u.hourangle, u.deg),
    )
    jpl_sb["ra"] = coord.ra.deg
    jpl_sb["dec"] = coord.dec.deg
    jpl_sb["V_mag"] = jpl_sb["Visual magnitude (V)"].replace("n.a.", np.nan)
    jpl_sb["V_mag"] = [
        float(x) if not x.endswith("N") else float(x[:-1]) for x in jpl_sb["V_mag"]
    ]
    jpl_sb['RA rate ("/h)'] = jpl_sb['RA rate ("/h)'].astype(float)
    jpl_sb['Dec rate ("/h)'] = jpl_sb['Dec rate ("/h)'].astype(float)
    jpl_sb["name"] = jpl_sb["Object name"].apply(lambda x: x.split("(")[0].strip())
    jpl_sb["id"] = jpl_sb["Object name"].apply(
        lambda x: x.split("(")[1][:-1].strip()
        if len(x.split("(")) > 1
        else x.split("(")[0].strip()
    )
    return jpl_sb


def get_asteroid_table(
    edge1: SkyCoord,
    edge2: SkyCoord,
    sector: int = 1,
    camera: int = 1,
    ccd: int = 1,
    maglim: float = 30,
    save: bool = True,
):
    scc_str = f"s{sector:04}-{camera}-{ccd}"
    jpl_sbi_file = f"{os.path.dirname(PACKAGEDIR)}/data/jpl_sbi/jpl_small_bodies_tess_{scc_str}_results.csv"

    if os.path.isfile(jpl_sbi_file):
        print("Loading from CSV file...")
        jpl_sb = pd.read_csv(jpl_sbi_file)
    else:
        jpl_sb = query_jpl_sbi(
            SkyCoord(ra_2d.min() * u.deg, dec_2d.min() * u.deg, frame='icrs'),
            SkyCoord(ra_2d.max() * u.deg, dec_2d.max() * u.deg, frame='icrs'),
            obstime=date_obs.mean().jd,
            maglim=maglim,
        )
        if save:
            print(f"Saving to {jpl_sbi_file}")
            jpl_sb.to_csv(jpl_sbi_file)
    return jpl_sb


def get_sector_dates(sector: int = 1):
    """
    Get sector observation dates (start, end) as a astropy Time object.
    """
    sector_date = pd.read_csv(
        f"{os.path.dirname(PACKAGEDIR)}/data/support/TESS_FFI_observation_times.csv"
    ).query(f"Sector == {sector}")
    if len(sector_date) == 0:
        raise ValueError(f"Sector {sector} not in data base.")

    return Time([sector_date.iloc[0]["Start Time"], sector_date.iloc[-1]["Start Time"]])


def run_FFI(
    sector: int = 1,
    camera: int = 1,
    ccd: int = 1,
    cutout_size: int = 50,
    plot: bool = True,
    maglim: float = 22,
):

    # get FFI file path
    sector_date = pd.read_csv(
        f"{os.path.dirname(PACKAGEDIR)}/data/support/TESS_FFI_observation_times.csv"
    ).query(f"sector == {sector}")
    path = "https://archive.stsci.edu/missions/tess/ffi/"
    ffi_file = (
        f"{path}/s{sector:04}/{date[:4]}/{date[4:7]}/{camera}-{ccd}"
        f"/tess{date}-{scc_str}-0214-s_ffic.fits"
    )

    # read FFI to get time, WCS and header
    with fits.open(ffi_file, mode="readonly") as hdulist:
        wcs = WCS(hdulist[1])
        date_obs = Time([hdulist[1].header["DATE-OBS"], hdulist[1].header["DATE-END"]])
        ffi_header = hdulist[1].header

    # load flux, columns and rows
    col_2d, row_2d, f2d = _load_ffi_image(
        "TESS",
        fits_file,
        1,
        None,
        [0, 0],
        return_coords=True,
    )
    # convert to ra, dec
    ra_2d, dec_2d = wcs.all_pix2world(
        np.vstack([col_2d.ravel(), row_2d.ravel()]).T, 0.0
    ).T
    ra_2d = ra_2d.reshape(col_2d.shape)
    dec_2d = dec_2d.reshape(col_2d.shape)

    # get low-res asteroid table from
    jpl_df = get_asteroid_table(
        SkyCoord(ra_2d.min() * u.deg, dec_2d.min() * u.deg, frame='icrs'),
        SkyCoord(ra_2d.max() * u.deg, dec_2d.max() * u.deg, frame='icrs'),
        sector=sector,
        camera=camera,
        ccd=ccd,
    )
    if maglim <= 24:
        asteroid_df = jpl_df.query("V_mag <= 18")

    global_name = f"tess-ffi_s{sector:04}-{camera}-{ccd}"
    if plot:
        fig_path = f"{os.path.dirname(PACKAGEDIR)}/data/figures"
        if not os.path.isdir(fig_path):
            os.mkdirs(fig_path)

        file_name = f"{fig_path}/{global_name}_diagnostic_plot.pdf"
        with PdfPages(file_name) as pages:
            fig_dist, ax = plt.subplots(1, 3, figsize=(16, 3))
            fig_dist.suptitle(
                f"Asteroids Distributions in Sector {sector} Camera {camera} CCD {ccd}"
            )
            ax[0].hist(jpl_df["V_mag"].values, bins=50)
            ax[0].hist(asteroid_df["V_mag"].values, bins=50)
            ax[0].set_xlabel("Visual Magnitude [mag]")
            ax[0].set_ylabel("N")
            ax[1].hist(jpl_df["RA rate (\"/h)"].values, bins=50, log=True)
            ax[1].hist(asteroid_df["RA rate (\"/h)"].values, bins=50, log=True)
            ax[1].set_xlabel("R.A. rate [\"/h]")
            ax[1].set_ylabel("N")
            ax[2].hist(jpl_df["Dec rate (\"/h)"].values, bins=50, log=True)
            ax[2].hist(asteroid_df["Dec rate (\"/h)"].values, bins=50, log=True)
            ax[2].set_xlabel("Dec. rate [\"/h]")
            ax[2].set_ylabel("N")

            FigureCanvasPdf(fig_dist).print_figure(pages)


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
