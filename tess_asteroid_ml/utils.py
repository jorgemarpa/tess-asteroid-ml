import numpy as np
import fitsio
from astropy.stats import sigma_clip


def power_find(n):
    """
    Function to decompose a number into power of 2.

    Parameters
    ----------
    n: int
        Number to be decompose.

    Returns
    -------
    result: list
        List of power of 2.
    """
    result = []
    binary = bin(n)[:1:-1]
    for x in range(len(binary)):
        if int(binary[x]):
            result.append(2 ** x)

    return result


def in_cutout(cutout_col, cutout_row, asteroid_col, asteroid_row, tolerance=2):
    """
    Auxiliar function to check if an asteroid track is inside a cutout.

    Parameters
    ----------
    cutout_col: numpy.ndarray
        Array with cutout column pixel numbers.
    cutout_row: numpy.ndarray
        Array with cutout row pixel numbers.
    asteroid_col: numpy.ndarray
        Array with asteroid track column pixel ppsition.
    asteroid_row: numpy.ndarray
        Array with asteroid track row pixel ppsition.
    tolerance: int
        Pixel tolerance.

    Returns
    -------
    is_in: bool
        Boolean if an asteroid is in the cutout or not.
    """
    is_in = (
        (asteroid_col >= cutout_col.min() - tolerance)
        & (asteroid_col <= cutout_col.max() + tolerance)
        & (asteroid_row >= cutout_row.min() - tolerance)
        & (asteroid_row <= cutout_row.max() + tolerance)
    )
    return is_in.any()


def load_ffi_image(
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


def fit_background(cube, cadenceno: list=[], polyorder: int = 1, positive_flux: bool = False):
    """Fit a simple 2d polynomial background to a TPF or cutout

    Parameters
    ----------
    cube: numpy.ndarray
        data cube of dimention [T, H, W]
    cadenceno: list, numpy.array
        Array with time index or cadence number
    polyorder: int
        Polynomial order for the model fit.
    positive_flux: bool
        Avoid negative numbers after removing background by adding a zero point value.

    Returns
    -------
    cube : np.ndarray
        Data cube woht background removed.
    """

    if not isinstance(cube, np.ndarray):
        raise ValueError("Input is not a Numpy ND Array")
    
    if len(cube.shape) != 3:
        raise ValueError("Input has to have 3 dimensions [T, H, W]")

    if (np.prod(cube.shape[1:]) < 100) | np.any(np.asarray(cube.shape[1:]) < 6):
        raise ValueError("TPF too small. Use a bigger cut out.")

    # Grid for calculating polynomial
    R, C = np.mgrid[: cube.shape[1], : cube.shape[2]].astype(float)
    R -= cube.shape[1] / 2
    C -= cube.shape[2] / 2

    def func(data):
        # Design matrix
        A = np.vstack(
            [
                R.ravel() ** idx * C.ravel() ** jdx
                for idx in range(polyorder + 1)
                for jdx in range(polyorder + 1)
            ]
        ).T

        # Median star image
        m = np.median(data, axis=0)
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
        f = np.vstack(data.transpose([1, 2, 0]))
        ws = np.linalg.solve(A.T.dot(A), A.T.dot(f))

        # Build a model that is just the polynomial
        model = (
            (A[:, :-1].dot(ws[:-1]))
            .reshape((data.shape[1], data.shape[2], data.shape[0]))
            .transpose([2, 0, 1])
        )
        # model += bkg0
        return model

    # Break point for TESS orbit
    if len(cadenceno) > 0:
        b = (
            np.where(np.diff(cadenceno) == np.diff(cadenceno).max())[0][0]
            + 1
        )
        # Calculate the model for each orbit, then join them
        bkg_model = np.vstack([func(aux) for aux in [cube[:b], cube[b:]]])
    else:
        bkg_model = func(cube)

    cube -= bkg_model.reshape(cube.shape[0], cube.shape[1], cube.shape[2])
    if positive_flux:
        cube += np.abs(np.floor(np.min(cube)))
    return cube
