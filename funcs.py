def check_position(ph, L, R):
    """ Checks whether a phasepoint is:
    - in the interval [L, R] : M
    - left of L              : L
    - right of R             : R

    Parameters
    ----------
    ph : tuple (x, v) of floats
        Phasepoint to check
    L : float
        Left boundary of the interval
    R : float
        Right boundary of the interval

    Returns
    -------
    str
        String representing the condition of the phasepoint

    """
    return "M" if L <= ph[0] <= R else "L" if ph[0] < L else "R"

