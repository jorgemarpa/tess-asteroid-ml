import diskcache
import numpy as np
import pandas as pd
import lightkurve as lk
import os
import fitsio
from tqdm import tqdm
from . import log, PACKAGEDIR
from .utils import power_find, in_cutout

cache = diskcache.Cache(directory="~/.tess-asteroid-ml-cache")


class AsteroidTESScut:
    """
    Class to obtain a TESS FFI cut and extract relevant data (flux, positions, etc),
    create a pixel mask where asteroids should be present, and save the data in a
    convinient format for ML use.

    It uses `lightkurve` to query and download a TESScut given R.A. and Dec coordinates
    and a pixel size (default 50x50).
    Will implement using `Astrocut` in MAST's TIKE to access TESS FFIs quicker.
    """

    def __init__(
        self,
        target: str,
        sector: int = 1,
        cutout_size: int = 50,
        quality_bitmask=None,
        use_tike: bool = False,
    ):
        """
        Parameters
        ----------
        target: str
            Target coordinates in the format 'R.A. Dec' with units [deg, deg].
        sector: int
            TESS sector to donwload.
        cutout_size: int
            Size in pixels of the square cutout.
        quality_bitmask: str or int
            Quality bitmask used for downloading the TESS cut. See details in
            https://docs.lightkurve.org/reference/api/lightkurve.TessTargetPixelFile.html

        Attributes
        ----------
        sector: int
            TESS sector number
        camera: int
            TESS camera number
        ccd: int
            TESS ccd number
        time: numpy.ndarray
            Array with time extracted from the TESS FFI cut. Has units of [jd].
        flux: numpy.ndarray
            Array with flux extracted from the TESS FFI cut. Has units of [-e/s].
        cadenceno: numpy.ndarray
            Array with cadence number extracted from the TESS FFI cut.
        column: numpy.ndarray
            Array with pixel column extracted from the TESS FFI cut.
        row: numpy.ndarray
            Array with pixel row extracted from the TESS FFI cut.
        ra: numpy.ndarray
            Array with pixel R.A. extracted from the TESS FFI cut. Has units of [deg].
        dec: numpy.ndarray
            Array with pixel Dec. extracted from the TESS FFI cut. Has units of [deg].
        cutout_size: int
            Cutout size in pixels
        ntimes: into
            Number of times/frames in the data.
        npixels: int
            Number of pixels in the cutout.
        tpf: lightkurve.TargetPixelFile
            TPF.
        """
        self.sector = sector
        if cutout_size >= 100:
            raise ValueError("Cutut size must be < 100 pix")
        self.cutout_size = cutout_size

        print("Querying TESScut")
        self.target_str = target
        if use_tike:
            raise NotImplementedError
        else:
            tpf = tpf = lk.search_tesscut(target, sector=self.sector).download(
                cutout_size=(self.cutout_size, self.cutout_size),
                quality_bitmask=quality_bitmask,
            )
        self.camera = tpf.camera
        self.ccd = tpf.ccd
        self.time = tpf.time.jd
        self.ntimes = self.time.shape[0]
        self.flux = tpf.flux.value
        self.flux = self.flux.reshape(self.ntimes, self.cutout_size * self.cutout_size)
        self.npixels = self.flux.shape[1]
        self.quality_mask = tpf.quality_mask
        self.cadenceno = tpf.cadenceno
        self.row, self.column = np.mgrid[
            tpf.row : tpf.row + tpf.shape[1],
            tpf.column : tpf.column + tpf.shape[2],
        ]
        self.column = self.column.ravel()
        self.row = self.row.ravel()
        self.ra, self.dec = tpf.get_coordinates(cadence=0)
        self.ra = self.ra.ravel()
        self.dec = self.dec.ravel()
        self.tpf = tpf

    @property
    def shape(self):
        return f"N times, N pixels: {self.ntimes}, {self.npixels}"

    @property
    def flux_2d(self):
        return self.flux.reshape(self.ntimes, self.cutout_size, self.cutout_size)

    @property
    def column_2d(self):
        return self.column.reshape(self.cutout_size, self.cutout_size)

    @property
    def row_2d(self):
        return self.row.reshape(self.cutout_size, self.cutout_size)

    @property
    def ra_2d(self):
        return self.ra.reshape(self.cutout_size, self.cutout_size)

    @property
    def dec_2d(self):
        return self.dec.reshape(self.cutout_size, self.cutout_size)

    def get_pos_corrs(self):
        """
        Get poscorr for the cutout. Poscorrs are defined for TPFs only, not for TESS
        cutouts or the FFI.
        """
        raise NotImplementedError

    # @cache.memoize(expire=2.592e06)
    def get_CBVs(self, align: bool = True, interpolate: bool = True):
        """
        Get CBVs for the cutout. TESS CBVs have 8 components.

        Parameters
        ----------
        align: bool
            Align or not the CBVs to the TPF.
        interpolate: bool
            Interpolate or not the CBVs to the TPF. This is necessary for TESS cuts.

        Attributes
        ----------
        cbvs: numpy.ndarray
            Array with the CBVs, has shape of `(ntimes, n_components)`
        """
        self.cbvs = lk.correctors.load_tess_cbvs(
            sector=self.sector,
            camera=self.camera,
            ccd=self.ccd,
            cbv_type="MultiScale",
            band=2,
        )
        if align or interpolate:
            target_mask = self.tpf.create_threshold_mask(
                threshold=15, reference_pixel='center'
            )
            ffi_lc = self.tpf.to_lightcurve(aperture_mask=target_mask)
            if align:
                self.cbvs = self.cbvs.align(ffi_lc)
            if interpolate:
                self.cbvs = self.cbvs.interpolate(ffi_lc, extrapolate=False)
        self._cbvs = self.cbvs.copy()
        cbvs = self.cbvs[
            [x for x in self.cbvs.columns if x.startswith("VECTOR")]
        ].as_array()
        self.cbvs = cbvs.view(np.float64).reshape(cbvs.shape + (-1,))
        return

    def get_asteroid_mask(
        self,
        asteroid_track: pd.DataFrame,
        name: str = "asteroid_001",
        mask_type: str = "circular",
        mask_radius: int = 2,
    ):
        """
        Creates am integer pixel mask with ones where an asteroids are found according
        to `asteroid_track`. The mask uses a circular aperture mask with radius
        `mask_radius`.

        Parameters
        ----------
        asteroid_track: pandas.DataFrame
            Data frame with asteroid tracks, has to have a column/row columns with
            asteroid's positions at each frame. Number of times must match `self.ntimes`
        name: str
            Name of the asteroid.
        mask_type: str
            Type of aperture, `'circular'` for a simple circular aperture (default),
            `'psf'` for an aperture considering the PSF shape (not implemented yet).
        mask_radius: int
            Radius of the circular aperture in pixels.

        Attributes
        ----------
        asteroid_mask: numpy.ndarray
            Integer mask array of shape `(ntimes, npixels)` with values corresponding to
            the `asteroid_names`.
        asteroid_names: dict
            Dictionary with corresponding number (key) and asteroid name (value). These
            numbers are in the `asteroid_mask`.
        asteroid_time_idx: dict
            Dictionary with asteroid number (key) and time index array (values) where
            the asteroid is detected.
        """
        if mask_type != "circular":
            raise NotImplementedError
        if not isinstance(asteroid_track, pd.DataFrame):
            raise ValueError("Input table must be a pandas DataFrame with column/row")
        if len(asteroid_track) != self.ntimes:
            print("Asteroid track has less times than TESScut. Will do interpolation")
            raise NotImplementedError

        if hasattr(self, "asteroid_mask"):
            asteroid_number = 2 ** (len(self.asteroid_names))
        else:
            asteroid_number = 2 ** 0
            self.asteroid_mask = np.zeros_like(self.flux, dtype=int)
            self.asteroid_names = {}
        self.asteroid_names[asteroid_number] = name

        for i, (t, row) in tqdm(
            enumerate(asteroid_track.iterrows()), total=len(asteroid_track)
        ):
            is_in = in_cutout(self.column, self.row, row["column"], row["row"])
            if is_in:
                has_asteroid = np.where(
                    np.hypot(self.column - row["column"], self.row - row["row"])
                    < mask_radius
                )[0]
                self.asteroid_mask[i, has_asteroid] += asteroid_number

        self.asteroid_time_idx = {}
        multiple_asteroid = [
            x for x in np.unique(self.asteroid_mask) if len(power_find(x)) > 1
        ]
        for n in self.asteroid_names.keys():
            aux = np.where((self.asteroid_mask == n).any(axis=1))[0]
            if len(multiple_asteroid) > 0:
                id_with_ast = [x for x in multiple_asteroid if n in power_find(x)]
                aux_2 = np.hstack(
                    [
                        np.where((self.asteroid_mask == x).any(axis=1))[0]
                        for x in id_with_ast
                    ]
                )
                aux = np.concatenate([aux, aux_2])
            self.asteroid_time_idx[n] = np.sort(aux)

        return

    @cache.memoize(expire=2.592e06)
    def get_quaternions(self, align: bool = True):
        """
        Get quaterionions vectors corresponding to the sector/camera/ccd
        """
        # url = "https://archive.stsci.edu/missions/tess/engineering/"
        # path = f"{PACKAGEDIR}/data/eng"
        dir_list = os.listdir(path)
        # if (
        #     len(dir_list) == 0
        #     or not ([f"sector{selfsector:02}-quat.fits" in x for x in dir_list]).any()
        # ):
        #     print("Downloading engineering files from MAST")
        #     response = requests.get(url)
        #     for c in response.iter_lines():
        #         line = c.decode("ascii")
        #         if f"sector{self.sector:02}-quat.fits" in line:
        #             fname = line.split("\"")[5]
        #     wget.download(f"{url}/{fname}", out=f"{path}")
        # else:
        #     fname = dir_list[
        #         [f"sector{selfsector:02}-quat.fits" in x for x in dir_list]
        #     ]
        # quat_table = fitsio.read(f"{path}/{fname}", ext=self.camera)
        #
        # self.quat_time = quat_table["TIME"]
        # self.quat_time = quat_table["TIME"]
        return

    def save_data(self, output=None):
        """
        Saves data into a *.npz file.

        Parameters
        ----------
        output: str
            Name of the output file, if None a defualt name will be asign.
        """
        if output is None:
            output = (
                f"{os.path.dirname(PACKAGEDIR)}/data/tesscuts"
                f"/tess-cut_asteroid_data_s{self.sector:04}-{self.camera}-{self.ccd}"
                f"_{self.ra.mean():.4f}_{self.dec.mean():.4f}"
                f"_{self.cutout_size}x{self.cutout_size}pix.npz"
            )

        medatada = {
            "sector": self.sector,
            "camera": self.camera,
            "ccd": self.ccd,
            "exp_time_s": self.ffi_exp_time if hasattr(self, "ffi_exp_time") else None,
            "detected_asterids": self.asteroid_names,
        }

        np.savez(
            output,
            flux=self.flux,
            mask=self.asteroid_mask,
            time=self.time,
            cbvs=self.cbvs,
            column=self.column,
            row=self.row,
            medatada=medatada,
        )

        return
