import diskcache
import numpy as np
import pandas as pd
import lightkurve as lk
import os
import fitsio
from tqdm import tqdm
from . import log, PACKAGEDIR

cache = diskcache.Cache(directory="~/.tess-asteroid-ml-cache")


class AsteroidTESScut:
    def __init__(
        self,
        target: str,
        sector: int = 1,
        cutout_size: int = 50,
        quality_bitmask=None,
    ):
        self.sector = sector
        if cutout_size >= 100:
            raise ValueError("Cutut size must be < 100 pix")
        self.cutout_size = cutout_size

        print("Querying TESScut")
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
        self.column, self.row = np.mgrid[
            tpf.column : tpf.column + tpf.shape[2],
            tpf.row : tpf.row + tpf.shape[1],
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

    def get_pos_corrs(self):
        """
        Get poscorr for the cutout. Poscorrs are defined for TPFs only, not for TESS
        cutouts or the FFI.
        """
        return

    # @cache.memoize(expire=2.592e06)
    def get_CBVs(self, align=True, interpolate=False):
        """
        Get poscorr for the cutout. Poscorrs are defined for TPFs only, not for TESS
        cutouts or the FFI.
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

    def _in_cutout(self, asteroid_col, asteroid_row, tolerance=2):
        is_in = (
            (asteroid_col >= self.column.min() - tolerance)
            & (asteroid_col <= self.column.max() + tolerance)
            & (asteroid_row >= self.row.min() - tolerance)
            & (asteroid_row <= self.row.max() + tolerance)
        )
        return is_in.any()

    def get_asteroid_mask(
        self, asteroid_track, name="asteroid_001", mask_type="circular", mask_radius=2
    ):
        """
        Creates a 0/1 pixel mask with ones where an asteroid is found according to JPL
        Horizon databse.
        """
        if mask_type != "circular":
            raise NotImplementedError
        if not isinstance(asteroid_track, pd.DataFrame):
            raise ValueError("Input table must be a pandas DataFrame with column/row")
        if len(asteroid_track) != self.ntimes:
            print("Asteroid track has less times than TESScut. Will do interpolation")
            raise NotImplementedError

        if hasattr(self, "asteroid_mask"):
            asteroid_number = len(self.asteroid_names) + 1
        else:
            asteroid_number = 1
            self.asteroid_mask = np.zeros_like(self.flux, dtype=int)
            self.asteroid_names = []
        self.asteroid_names.append([[asteroid_number, name]])

        for i, (t, row) in tqdm(
            enumerate(asteroid_track.iterrows()), total=len(asteroid_track)
        ):
            is_in = self._in_cutout(row["column"], row["row"])
            if is_in:
                has_asteroid = np.where(
                    np.hypot(self.column - row["column"], self.row - row["row"])
                    < mask_radius
                )[0]
                self.asteroid_mask[i, has_asteroid] = asteroid_number
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
        if output is None:
            output = (
                f"{os.path.dirname(PACKAGEDIR)}/data/tesscuts"
                f"/tess-cut_asteroid_data_s{self.sector:04}-{camera}-{ccd}.npz"
            )

        np.savez(
            output,
            flux=self.flux,
            mask=self.asteroid_mask,
            time=self.time,
            cbvs=self.cbvs,
            column=self.column,
            row=self.row,
        )

        return
