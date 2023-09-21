import numpy as np
import fitsio


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
