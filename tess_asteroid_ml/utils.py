import numpy as np


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
