import os
import argparse
import tempfile
import s3fs

# import nest_asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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

import tess_cloud
from tess_ephem import TessEphem
from tess_asteroid_ml import PACKAGEDIR
from tess_asteroid_ml.utils import in_cutout, load_ffi_image


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
    print(f"JPL SBI found {len(jpl_sb)} asteroids with V < {maglim} in {scc_str}")
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


def get_FFI_name(
    sector: int = 1, camera: int = 1, ccd: int = 0, correct=True, provider="mast"
):
    if ccd == 0:
        files = [
            tess_cloud.list_images(sector=sector, camera=1, ccd=ccd, author="spoc")
            for ccd in range(1, 5)
        ]
        frame = len(files[0]) // 2
        file_name = [x[frame].filename for x in files]

    else:
        files = tess_cloud.list_images(sector=sector, camera=1, ccd=ccd, author="spoc")
        frame = len(files) // 2
        file_name = files[frame].filename

    if provider == "mast":
        root_path = "https://archive.stsci.edu/missions/tess"
    elif provider == "aws":
        root_path = "s3://stpubdata/tess/public"

    aux = []
    for fn in file_name:
        date_o = fn[4:17]
        date_n = str(int(date_o) - 1)
        yyyy = date_o[:4]
        ddd = date_o[4:7]
        if correct:
            fn = fn.replace(date_o, date_n)
        camera = fn.split("-")[2]
        ccd = fn.split("-")[3]
        dir_path = f"ffi/s{sector:04}/{yyyy}/{ddd}/{camera}-{ccd}"
        aux.append(f"{root_path}/{dir_path}/{fn}")

    file_name = aux

    return file_name


def get_data_from_files(file_list, provider="mast"):
    r_min = 0
    r_max = 2048
    c_min = 44
    c_max = 2092
    row_2d, col_2d = np.mgrid[r_min:r_max, c_min:c_max]
    ffi_headers = []
    ffi_flux = []
    ffi_ra_2d, ffi_dec_2d = [], []
    for file in file_list:

        if provider == "mast":
            ffi_headers.append(fits.getheader(file))
            ffi_flux.append(fits.getdata(file)[r_min:r_max, c_min:c_max])
            wcs = WCS(fits.getheader(file, ext=1))
        else:
            fs = s3fs.S3FileSystem(anon=True)
            with tempfile.NamedTemporaryFile(mode="wb") as tmp:
                f = fs.get(file, tmp.name)
                ffi_headers.append(fits.getheader(tmp.name))
                ffi_flux.append(fits.getdata(tmp.name)[r_min:r_max, c_min:c_max])
                wcs = WCS(fits.getheader(tmp.name, ext=1))

        ra_2d, dec_2d = wcs.all_pix2world(
            np.vstack([col_2d.ravel(), row_2d.ravel()]).T, 0.0
        ).T
        ffi_ra_2d.append(ra_2d.reshape(col_2d.shape))
        ffi_dec_2d.append(dec_2d.reshape(col_2d.shape))

    return ffi_headers, ffi_flux, col_2d, row_2d, ffi_ra_2d, ffi_dec_2d


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
    lowres_time = sector_dates[0] + np.arange(0, days, 1.0)

    track_file_root = f"{os.path.dirname(PACKAGEDIR)}/data/jpl/tracks/sector{sector:04}"
    if not os.path.isdir(track_file_root):
        os.makedirs(track_file_root)

    print(f"Will find asteroid tracks as check if are on FFI...")
    sb_ephems = {}
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
                pass
            try:
                te = TessEphem(
                    row['name'],
                    start=sector_dates[0],
                    stop=sector_dates[-1],
                    step="12H",
                    id_type="smallbody",
                )
                name_ok = row['name']
            except ValueError:
                continue
            # predict asteroid position in the sector with low res
            try:
                ephems_aux = te.predict(
                    time=lowres_time,
                    aberrate=True,
                    verbose=True,
                )
            except SystemExit:
                print(f"`tess_stars2px` failed for {name_ok}. Continuing...")
                continue
            # filter by camera/ccd, if both 0 will save the full sector
            if camera != 0 and ccd != 0:
                ephems_aux = ephems_aux.query(f"camera == {camera} and ccd == {ccd}")
            if len(ephems_aux) == 0:
                continue
            # predict with high-res
            if do_highres and predict_times is not None:
                ephems_aux = te.predict(
                    time=predict_times,
                    aberrate=True,
                    verbose=True,
                )
            ephems_aux = ephems_aux.reset_index()
            ephems_aux["time"] = [x.jd for x in ephems_aux["time"].values]
            ephems_aux.to_feather(track_file)

        sb_ephems[k] = ephems_aux
    return sb_ephems


def create_FFI_asteroid_database(
    sector: int = 1,
    camera: int = 1,
    ccd: int = 1,
    plot: bool = True,
    maglim: float = 22,
    provider: str = "mast",
):

    # get FFI file path
    sector_dates = get_sector_dates(sector=sector)
    ffi_file = get_FFI_name(sector=sector, camera=camera, ccd=ccd, provider=provider)
    print(ffi_file)

    # read FFI to get time, WCS and header
    ffi_header, f2d, col_2d, row_2d, ra_2d, dec_2d = get_data_from_files(
        ffi_file, provider=provider
    )
    ffi_date = Time([ffi_header[0]["DATE-OBS"], ffi_header[0]["DATE-END"]])

    # get asteroid table from JPL SBI for Sector/Camera/CCD
    jpl_df = get_asteroid_table(
        SkyCoord(np.min(ra_2d) * u.deg, np.min(dec_2d) * u.deg, frame='icrs'),
        SkyCoord(np.max(ra_2d) * u.deg, np.max(dec_2d) * u.deg, frame='icrs'),
        sector=sector,
        camera=camera,
        ccd=ccd,
        date_obs=ffi_date.mean().jd,
    )
    # filter bright ateroids, 30 is the mag limit in the original JPL query
    if maglim <= 30:
        asteroid_df = jpl_df.query(f"V_mag <= {maglim}")

    time = get_sector_time_array(
        f"{ra_2d[0][1000, 900]:.5f} {dec_2d[0][1000, 900]:.5f}",
        sector=sector,
        camera=camera,
        ccd=camera,
        tike=False,
    )

    # get tracks of asteroids on the FFI between observing dates
    asteroid_tracks = get_asteroids_in_FFI(
        asteroid_df,
        get_sector_dates(sector=sector),
        do_highres=True,
        predict_times=time,
        sector=sector,
        camera=0,
        ccd=0,
    )
    print(f"Total asteroids (V < {maglim}) in FFI: {len(asteroid_tracks)}")

    global_name = f"tess-ffi_s{sector:04}-{camera}-{ccd}"
    if plot:
        fig_path = f"{os.path.dirname(PACKAGEDIR)}/data/figures"
        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)

        file_name = f"{fig_path}/{global_name}_diagnostic_plot.pdf"
        print(f"Saving figures to {file_name}")
        with PdfPages(file_name) as pages:
            fig_dist, ax = plt.subplots(1, 3, figsize=(16, 3), constrained_layout=True)
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

            if ccd > 0:
                fig_ima = plt.figure(figsize=(7, 7), constrained_layout=True)
                plt.title(
                    f"Asteroid tracks in Sector {sector} Camera {camera} CCD {ccd}"
                )
                plt.pcolormesh(
                    col_2d,
                    row_2d,
                    f2d[0],
                    vmin=400,
                    vmax=2000,
                    cmap="Greys_r",
                    rasterized=True,
                )

                for k, val in asteroid_tracks.items():
                    if len(val) == 0:
                        continue
                    n = 50 if len(val.column) > 1000 else 1
                    plt.plot(
                        val.column[::n],
                        val.row[::n],
                        marker=".",
                        linewidth=1,
                        markersize=0.1,
                        rasterized=True,
                    )
                    # if k == 1: break

                plt.xlim(col_2d.min() - 10, col_2d.max() + 10)
                plt.ylim(row_2d.min() - 10, row_2d.max() + 10)
                plt.gca().set_aspect('equal')
                plt.xlabel("Pixel Column")
                plt.ylabel("Pixel Row")
                FigureCanvasPdf(fig_ima).print_figure(pages)
                plt.close()
            else:
                fig_ima, ax = plt.subplots(
                    2,
                    2,
                    figsize=(7, 7),
                    sharex=True,
                    sharey=True,
                    constrained_layout=True,
                )
                fig_ima.suptitle(f"Asteroid tracks in Sector {sector} Camera {camera}")
                for i, axis in enumerate(ax.ravel()):
                    axis.pcolormesh(
                        col_2d,
                        row_2d,
                        f2d[i],
                        # vmin=400,
                        # vmax=2000,
                        cmap="Greys_r",
                        norm=colors.SymLogNorm(
                            linthresh=1000, vmin=100, vmax=2500, base=10
                        ),
                        rasterized=True,
                    )

                    for k, val in asteroid_tracks.items():
                        val = val.query(f"ccd == {i + 1}")
                        if len(val) == 0:
                            continue
                        # n = 50 if len(val.column) > 1000 else 1
                        axis.plot(
                            val.column,
                            val.row,
                            ".-",
                            ms=0.2,
                            lw=0.1,
                            rasterized=True,
                        )
                    # if k == 1: break

                ax[0, 0].set_xlim(col_2d.min() - 10, col_2d.max() + 10)
                ax[0, 0].set_ylim(row_2d.min() - 10, row_2d.max() + 10)
                ax[0, 0].set_aspect("equal", adjustable="box")
                ax[1, 0].set_xlabel("Pixel Column")
                ax[1, 1].set_xlabel("Pixel Column")
                ax[0, 0].set_ylabel("Pixel Row")
                ax[1, 0].set_ylabel("Pixel Row")
                FigureCanvasPdf(fig_ima).print_figure(pages)
                plt.close()

    print("Done!")
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
    args = parser.parse_args()
    create_FFI_asteroid_database(
        sector=args.sector,
        camera=args.camera,
        ccd=args.ccd,
        plot=False,
    )
