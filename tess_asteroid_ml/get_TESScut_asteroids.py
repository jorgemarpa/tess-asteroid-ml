import os
import argparse
import tempfile
import requests

# import nest_asyncio
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm
from astrocut import CutoutFactory
from typing import Optional
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages

from astroquery.jplhorizons import Horizons
from sbident import SBIdent

from tess_ephem import TessEphem
from tess_asteroid_ml import *
from tess_asteroid_ml.utils import in_cutout, load_ffi_image


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

    # 1AU in km
    au = (1 * u.au).to(u.km).value
    # TESS state vector
    tess = Horizons(id='-95', location='500', epochs=obstime, id_type=None).vectors(
        refplane='earth'
    )
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
    edge1: SkyCoord = None,
    edge2: SkyCoord = None,
    date_obs: Time = None,
    sector: int = 1,
    camera: int = 1,
    ccd: int = 1,
    maglim: float = 30,
    save: bool = True,
):
    scc_str = f"s{sector:04}-{camera}-{ccd}"
    jpl_sbi_file = f"{os.path.dirname(PACKAGEDIR)}/data/jpl/jpl_small_bodies_tess_{scc_str}_catalog.csv"
    if os.path.isfile(jpl_sbi_file):
        print(f"Loading from CSV file: {jpl_sbi_file}")
        jpl_sb = pd.read_csv(jpl_sbi_file)
    else:
        jpl_sb = query_jpl_sbi(
            edge1,
            edge2,
            obstime=date_obs,
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

    return Time(
        [
            sector_date.iloc[0]["Start Time"],
            sector_date.iloc[1]["Start Time"],
            sector_date.iloc[-1]["End Time"],
        ],
        format='iso',
    )


def get_FFI_name(sector: int = 1, camera: int = 1, ccd: int = 1):
    """
    Finds an FFI url in MAST archive to load as frame of reference.
    """
    root_path = "https://archive.stsci.edu/missions/tess/ffi"
    sector_dates = get_sector_dates(sector=sector)
    yyyy, ddd, hh, mm, ss = sector_dates[1].yday.split(":")
    dir_path = f"{root_path}/s{sector:04}/{yyyy}/{ddd}/{camera}-{ccd}"
    response = requests.get(dir_path)
    for k in response.iter_lines():
        if "ffic.fits" in k.decode():
            ffi_name = k.decode().split("\"")[5]
            break

    ffi_path = f"{dir_path}/{ffi_name}"
    return ffi_path


def get_sector_time_array(
    target, sector: int = 1, camera: int = 1, ccd: int = 1, tike=False
):
    """
    Returns the time array for a given sector. It uses a TESS cut.
    """
    if tike:
        factory = CutoutFactory()
        cube_file = f"s3://stpubdata/tess/public/mast/tess-s{sector:04d}-{camera}-{ccd}-cube.fits"
        with tempfile.NamedTemporaryFile(mode="wb") as tmp:
            tpf_path = factory.cube_cut(
                cube_file=cube_file,
                coordinates=target,
                cutout_size=10,
                target_pixel_file=tmp.name,
            )
            tpf = lk.read(tpf_path)
    else:
        tpf = tpf = lk.search_tesscut(target, sector=sector).download(
            cutout_size=(10, 10), quality_bitmask=None
        )
    return tpf.time


def get_asteroids_in_FFI(
    df,
    ffi_row: Optional[npt.ArrayLike] = None,
    ffi_col: Optional[npt.ArrayLike] = None,
    sector_dates: Optional[Time] = None,
    do_highres: bool = False,
    predict_times=None,
    sector: int = 1,
    camera: int = 0,
    ccd: int = 0,
):
    """
    Creates a dictionary with asteroids track if the asteroid is observed on the FFI.
    It does a query to JPL via `tess-ephem` with sectors start/end dates and predicts
    the asteroid location in pixel coordinates.
    """

    if len(sector_dates) < 2:
        raise ValueError("Please provide at least two observing dates (start and end)")

    # low-res 1-day interval
    days = np.ceil((sector_dates[-1] - sector_dates[0]).sec / (60 * 60 * 24))
    lowres_time = sector_dates[0] + np.arange(-1, days + 1, 1.0)

    sb_ephems = {}
    track_file_root = f"{os.path.dirname(PACKAGEDIR)}/data/jpl/tracks/sector{sector:04}"
    if not os.path.isfile(track_file_root):
        os.mkdirs(track_file_root)

    for k, row in tqdm(df.iterrows(), total=len(df), desc="JPL query"):
        # if k > 50:
        #     break
        # read file with asteroid track from disk if exists
        track_file = (
            f"{track_file_root}/"
            f"tess-ffi_s{sector:04}-{camera}-{ccd}_{row['id'].replace(' ', '-')}_"
            f"{'hi' if do_highres else 'lo'}res.feather"
        )
        if os.path.isfile(track_file):
            ephems_aux = pd.read_feather(track_file)
        # query JPL if not
        else:
            # query JPL to get asteroid track within sector times every 12h
            try:
                te = TessEphem(
                    row['id'],
                    start=sector_dates[0],
                    stop=sector_dates[-1],
                    step="12H",
                    id_type="smallbody",
                )
                name_ok = row['id']
            except ValueError:
                te = TessEphem(
                    row['name'],
                    start=sector_dates[0],
                    stop=sector_dates[-1],
                    step="12H",
                    id_type="smallbody",
                )
                name_ok = row['name']
            # predict asteroid position in the sector with low res
            ephems_aux = te.predict(
                time=lowres_time,
                aberrate=True,
                verbose=True,
            )
            # filter by camera/ccd, if both 0 will save the full sector
            if camera != 0 and ccd != 0:
                ephems_aux = ephems_aux.query(f"camera == {camera} and ccd == {ccd}")
            if len(ephems_aux) == 0:
                continue
            # predict with high-res
            if do_highres and predict_times is not None:
                is_in = in_cutout(
                    ffi_col, ffi_row, ephems_aux.column.values, ephems_aux.row.values
                )
                # check if track is on the FFI
                if is_in:
                    ephems_aux = te.predict(
                        time=predict_times,
                        aberrate=True,
                        verbose=True,
                    )
            ephems_aux = ephems_aux.reset_index()
            ephems_aux["time"] = [x.jd for x in ephems_aux["time"].values]
            ephems_aux.to_feather(track_file)
        sb_ephems[k] = ephems_aux
    else:
        return sb_ephems


def create_FFI_asteroid_database(
    sector: int = 1,
    camera: int = 1,
    ccd: int = 1,
    plot: bool = True,
    maglim: float = 22,
):

    # get FFI file path
    sector_dates = get_sector_dates(sector=1)
    yyyy, ddd, hh, mm, ss = sector_dates.mean().yday.split(":")
    path = "https://archive.stsci.edu/missions/tess/ffi/"
    ffi_file = get_FFI_name(sector=sector, camera=camera, ccd=ccd)
    print(ffi_file)

    # read FFI to get time, WCS and header
    with fits.open(ffi_file, mode="readonly") as hdulist:
        wcs = WCS(hdulist[1])
        ffi_date = Time([hdulist[1].header["DATE-OBS"], hdulist[1].header["DATE-END"]])
        ffi_header = hdulist[1].header

    # load flux, columns and rows
    col_2d, row_2d, f2d = load_ffi_image(
        "TESS",
        ffi_file,
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

    # get asteroid table from JPL SBI for Sector/Camera/CCD
    jpl_df = get_asteroid_table(
        SkyCoord(ra_2d.min() * u.deg, dec_2d.min() * u.deg, frame='icrs'),
        SkyCoord(ra_2d.max() * u.deg, dec_2d.max() * u.deg, frame='icrs'),
        sector=sector,
        camera=camera,
        ccd=ccd,
        date_obs=ffi_date.mean().isot,
    )
    # filter bright ateroids, 30 is the mag limit in the original JPL query
    if maglim <= 30:
        asteroid_df = jpl_df.query(f"V_mag <= {maglim}")

    time = get_sector_time_array(
        f"{ra_2d[1000, 900]:.5f} {dec_2d[1000, 900]:.5f}",
        sector=sector,
        camera=camera,
        ccd=camera,
        tike=False,
    )

    # get tracks of asteroids on the FFI between observing dates
    asteroid_tracks = get_asteroids_in_FFI(
        asteroid_df,
        row_2d.ravel(),
        col_2d.ravel(),
        get_sector_dates(sector=sector),
        do_highres=True,
        predict_times=time,
        sector=sector,
        camera=camera,
        ccd=ccd,
    )
    print(f"Total asteroids (V < {maglim}) in FFI: {len(asteroid_tracks)}")

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
            plt.close()

            fig_ima = plt.figure(figsize=(7, 7))
            plt.title(f"Asteroid tracks in Sector {sector} Camera {camera} CCD {ccd}")
            plt.pcolormesh(
                col_2d,
                row_2d,
                f2d,
                vmin=400,
                vmax=2000,
                cmap="Greys_r",
                rasterized=True,
            )

            for k, val in asteroid_tracks.items():
                if len(val) == 0:
                    continue
                n = 50 if len(val.column) > 1000 else 1
                plt.plot(val.column[::n], val.row[::n], ".-", ms=0.5, rasterized=True)
                # if k == 1: break

            plt.xlim(col_2d.min() - 10, col_2d.max() + 10)
            plt.ylim(row_2d.min() - 10, row_2d.max() + 10)
            plt.gca().set_aspect('equal')
            plt.xlabel("Pixel Column")
            plt.ylabel("Pixel Row")
            FigureCanvasPdf(fig_ima).print_figure(pages)
            plt.close()

    print("Done!")
    return


def create_cutout_data(
    sector: int = 1,
    camera: int = 1,
    ccd: int = 1,
    cutout_size: int = 50,
    maglim: float = 22,
):
    jpl_df = get_asteroid_table(
        SkyCoord(ra_2d.min() * u.deg, dec_2d.min() * u.deg, frame='icrs'),
        SkyCoord(ra_2d.max() * u.deg, dec_2d.max() * u.deg, frame='icrs'),
        sector=sector,
        camera=camera,
        ccd=ccd,
    )
    if maglim <= 24:
        asteroid_df = jpl_df.query("V_mag <= 18")

    asteroid_tracks = get_asteroids_in_FFI(
        asteroid_df,
        row_2d.ravel(),
        col_2d.ravel(),
        get_sector_dates(sector=sector),
        do_highres=True,
        predict_times=None,
        sector=sector,
        camera=camera,
        ccd=ccd,
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
        "--cutout-size",
        dest="cutout_size",
        type=int,
        default=None,
        help="Cutout size in pixels",
    )
    parser.add_argument(
        "--make-db",
        dest="make_db",
        action="store_true",
        default=False,
        help=(
            "Creates the asteroid tracks data base."
            "You should run this only one time per Sector/Camera/CCD"
        ),
    )
    args = parser.parse_args()
    if args.make_db:
        create_FFI_asteroid_database(
            sector=args.sector,
            camera=args.camera,
            ccd=args.ccd,
        )
