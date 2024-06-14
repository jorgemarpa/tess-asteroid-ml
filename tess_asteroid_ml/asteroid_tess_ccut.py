import diskcache
import numpy as np
import pandas as pd
import lightkurve as lk
import os
from . import log, PACKAGEDIR
from .utils import power_find, in_cutout
from scipy.interpolate import CubicSpline
from astropy.stats import sigma_clip

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
        tpf: lk.TessTargetPixelFile = None,
        target: str = "",
        sector: int = 1,
        cutout_size: int = 50,
        quality_bitmask="default",
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
        if isinstance(tpf, lk.targetpixelfile.TargetPixelFile):
            self.tpf = tpf
        elif isinstance(target, str):
            print("Querying TESScut")
            if cutout_size >= 100:
                raise ValueError("Cutut size must be < 100 pix")
            self.tpf = lk.search_tesscut(target, sector=sector).download(
                cutout_size=(cutout_size, cutout_size),
                quality_bitmask=quality_bitmask,
            )
        self.quality_mask = lk.utils.TessQualityFlags.create_quality_mask(
            self.tpf.quality, bitmask=quality_bitmask
        )
        self.cadenceno = self.tpf.cadenceno.copy()
        # remove cadences with flux == 0s in all pixels
        self.quality_mask &= (self.tpf.flux == 0.0).sum(axis=-1).sum(axis=-1) == 0
        self.tpf = self.tpf[self.quality_mask]
        self.target_str = self.tpf.targetid
        self.sector = self.tpf.sector
        self.cutout_size = self.tpf.shape[1]
        self.camera = self.tpf.camera
        self.ccd = self.tpf.ccd
        self.time = self.tpf.time.jd
        self.ntimes = self.time.shape[0]
        self.flux = self.tpf.flux.value.reshape(
            self.ntimes, self.cutout_size * self.cutout_size
        )
        self.npixels = self.flux.shape[1]
        self.row, self.column = np.mgrid[
            self.tpf.row : self.tpf.row + self.tpf.shape[1],
            self.tpf.column : self.tpf.column + self.tpf.shape[2],
        ]
        self.column = self.column.ravel()
        self.row = self.row.ravel()
        self.ra, self.dec = self.tpf.get_coordinates(cadence=0)
        self.ra = self.ra.ravel()
        self.dec = self.dec.ravel()
        self.asteroid_mask = np.zeros_like(self.flux, dtype=np.int64)

    @property
    def shape(self):
        return f"N times, N pixels: ({self.ntimes}, {self.npixels})"

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

    @property
    def asteroid_mask_2d(self):
        if hasattr(self, "asteroid_mask"):
            return self.asteroid_mask.reshape(
                self.ntimes, self.cutout_size, self.cutout_size
            )

    def get_pos_corrs(self):
        """
        Get poscorr for the cutout. Poscorrs are defined for TPFs only, not for TESS
        cutouts or the FFI.
        """
        raise NotImplementedError

    # @cache.memoize(expire=2.592e06)
    def get_CBVs(self, align: bool = False, interpolate: bool = True):
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
        # self._cbvs = self.cbvs.copy()
        if align or interpolate:
            target_mask = self.tpf.create_threshold_mask(
                threshold=15, reference_pixel="center"
            )
            ffi_lc = self.tpf.to_lightcurve(aperture_mask=target_mask)
            if align:
                self.cbvs = self.cbvs.align(ffi_lc)
            if interpolate:
                self.cbvs = self.cbvs.interpolate(ffi_lc, extrapolate=False)
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
        mask_num_type: str = "byte",
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
            # print("Asteroid track has less times than TESScut. Will do interpolation")
            # interpolate to tess cut times using highres track
            _colf = CubicSpline(asteroid_track.time.values, asteroid_track.column)
            _rowf = CubicSpline(asteroid_track.time.values, asteroid_track.row)
            # mask with intersection of times between track as tess cut
            ti = np.where(self.time >= asteroid_track.time.values[0])[0][0]
            # if ti > 0:
            #     ti -= 1
            tf = np.where(self.time >= asteroid_track.time.values[-1])[0]
            if len(tf) > 0:
                tf = tf[0]
            else:
                tf = len(self.time)
            tmask = np.zeros_like(self.time, dtype=bool)
            tmask[ti:tf] = True
            # interpolate
            col_int = np.zeros_like(self.time)
            row_int = np.zeros_like(self.time)
            col_int[tmask] = _colf(self.time[tmask])
            row_int[tmask] = _rowf(self.time[tmask])
            # raise NotImplementedError
        else:
            # use values in track argument
            tmask = np.ones_like(self.time, dtype=bool)
            col_int = asteroid_track.column.values
            row_int = asteroid_track.row.values

        # check if Attributes exist if not create new
        if hasattr(self, "asteroid_names"):
            if mask_num_type == "byte":
                asteroid_number = 2 ** (len(self.asteroid_names))
            elif mask_num_type == "index" and isinstance(name, pd.Series):
                asteroid_number = name.name
            else:
                asteroid_number = (len(self.asteroid_names)) + 1
        else:
            if mask_num_type == "byte":
                asteroid_number = 2**0
            elif mask_num_type == "index" and isinstance(name, pd.Series):
                asteroid_number = name.name
            else:
                asteroid_number = 1
            # self.asteroid_mask = np.zeros_like(self.flux, dtype=int)
            self.asteroid_names = {}
        self.asteroid_names.update({asteroid_number: name})
        # iterate over times to fill up asteroid_mask
        for i, t in enumerate(tmask):
            # only use times within the track table
            if t:
                is_in = in_cutout(self.column, self.row, col_int[i], row_int[i])
                if is_in:
                    has_asteroid = np.where(
                        np.hypot(self.column - col_int[i], self.row - row_int[i])
                        < mask_radius
                    )[0]
                    if mask_num_type == "byte":
                        self.asteroid_mask[i, has_asteroid] += asteroid_number
                    else:
                        self.asteroid_mask[i, has_asteroid] = asteroid_number

        # dictionary with time idx for each asteroid
        self.asteroid_time_idx = {}
        if mask_num_type == "byte":
            multiple_asteroid = [
                x for x in np.unique(self.asteroid_mask) if len(power_find(x)) > 1
            ]
            for n in self.asteroid_names.keys():
                aux = np.where((self.asteroid_mask == n).any(axis=1))[0]
                if len(multiple_asteroid) > 0:
                    id_with_ast = [x for x in multiple_asteroid if n in power_find(x)]
                    if len(id_with_ast) > 0:
                        aux_2 = np.hstack(
                            [
                                np.where((self.asteroid_mask == x).any(axis=1))[0]
                                for x in id_with_ast
                            ]
                        )
                        aux = np.concatenate([aux, aux_2])
                self.asteroid_time_idx[n] = np.sort(aux)
        else:
            for n in self.asteroid_names.keys():
                self.asteroid_time_idx[n] = np.sort(
                    np.where((self.asteroid_mask == n).any(axis=1))[0]
                )

        return

    def get_quaternions_and_angles(self):
        """
        Get quaterionions and angle vectors corresponding to the sector/camera.
        Uses the cadence number and quality mask to match times.
        """
        ff = (
            f"{os.path.dirname(PACKAGEDIR)}/data/engineering/TESSVectors_S1-26_FFI"
            f"/TessVectors_S{self.sector:03}_C{self.camera}_FFI.csv"
        )
        vectors = pd.read_csv(ff, skiprows=44)

        if vectors.shape[0] != self.quality_mask.shape[0]:
            raise ValueError("Number of cadences do not match.")
        self.quaternions = vectors.loc[
            self.quality_mask, ["Quat1_Med", "Quat2_Med", "Quat3_Med", "Quat4_Med"]
        ].values
        self.earth_angle = vectors.loc[
            self.quality_mask, ["Earth_Camera_Angle", "Earth_Camera_Azimuth"]
        ].values
        self.moon_angle = vectors.loc[
            self.quality_mask, ["Moon_Camera_Angle", "Moon_Camera_Azimuth"]
        ].values
        return

    def save_data(self, output: str = None):
        """
        Saves data into a *.npz file.

        Parameters
        ----------
        output: str
            Name of the output file, if None a defualt name will be asign.
        """
        if output is None:
            output = (
                f"{os.path.dirname(PACKAGEDIR)}/data/asteroidcuts/sector{self.sector:04}"
                f"/tess-cut_asteroid_data_s{self.sector:04}-{self.camera}-{self.ccd}"
                f"_{self.ra.mean():.4f}_{self.dec.mean():.4f}"
                f"_{self.cutout_size}x{self.cutout_size}pix.npz"
            )
            if not os.path.isdir(os.path.dirname(output)):
                os.makedirs(os.path.dirname(output))

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

    def find_orbit_breaks(self):
        """
        Cuts sector data into two orbits
        """
        # finds the indx where orbit starts and ends
        dts = np.diff(self.time)
        self.breaks = np.where(dts >= 0.2)[0] + 1
        return

    def fit_background(self, polyorder: int = 1, positive_flux: bool = False):
        """Fit a simple 2d polynomial background to a TPF

        Parameters
        ----------
        tpf: lightkurve.TessTargetPixelFile
            Target pixel file object
        polyorder: int
            Polynomial order for the model fit.

        Returns
        -------
        model : np.ndarray
            Model for background with same shape as tpf.shape
        """

        if not isinstance(self.tpf, lk.TessTargetPixelFile):
            raise ValueError("Input a TESS Target Pixel File")

        if (np.prod(self.tpf.shape[1:]) < 100) | np.any(
            np.asarray(self.tpf.shape[1:]) < 6
        ):
            raise ValueError("TPF too small. Use a bigger cut out.")

        # Grid for calculating polynomial
        R, C = np.mgrid[: self.tpf.shape[1], : self.tpf.shape[2]].astype(float)
        R -= self.tpf.shape[1] / 2
        C -= self.tpf.shape[2] / 2

        def func(tpf):
            # Design matrix
            A = np.vstack(
                [
                    R.ravel() ** idx * C.ravel() ** jdx
                    for idx in range(polyorder + 1)
                    for jdx in range(polyorder + 1)
                ]
            ).T

            # Median star image
            m = np.median(tpf.flux.value, axis=0)
            # Remove background from median star image
            mask = ~sigma_clip(m, sigma=3).mask.ravel()
            # plt.imshow(mask.reshape(m.shape))
            bkg0 = A.dot(
                np.linalg.solve(A[mask].T.dot(A[mask]), A[mask].T.dot(m.ravel()[mask]))
            ).reshape(m.shape)

            m -= bkg0

            # Include in design matrix
            A = np.hstack([A, m.ravel()[:, None]])

            # Fit model to data, including a model for the stars
            f = np.vstack(tpf.flux.value.transpose([1, 2, 0]))
            ws = np.linalg.solve(A.T.dot(A), A.T.dot(f))

            # Build a model that is just the polynomial
            model = (
                (A[:, :-1].dot(ws[:-1]))
                .reshape((tpf.shape[1], tpf.shape[2], tpf.shape[0]))
                .transpose([2, 0, 1])
            )
            # model += bkg0
            return model

        # Break point for TESS orbit
        b = (
            np.where(np.diff(self.tpf.cadenceno) == np.diff(self.tpf.cadenceno).max())[
                0
            ][0]
            + 1
        )

        # Calculate the model for each orbit, then join them
        self.bkg_model = np.vstack([func(aux) for aux in [self.tpf[:b], self.tpf[b:]]])
        self.flux -= self.bkg_model.reshape(
            self.ntimes, self.cutout_size * self.cutout_size
        )
        if positive_flux:
            self.flux += np.abs(np.floor(np.min(self.flux)))
        return
