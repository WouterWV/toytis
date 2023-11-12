import os
import matplotlib.pyplot as plt
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

def plot_paths(paths, intfs=None, ax=None, start_ids=0, **kwargs):
    """ Plots the paths in the list paths, with optional interfaces intfs.
    
    Parameters
    ----------
    paths : list of :py:class:`Path` objects
        Paths to plot
        
    intfs : list of floats, optional
        Interfaces to plot

    """
    if start_ids == 0:
        start_ids = [0 for _ in paths]
    elif start_ids == "staggered":
        start_ids = [0]
        for path in paths[:-1]:
            start_ids.append(start_ids[-1] + len(path.phasepoints))
    assert len(start_ids) == len(paths)
    if ax is None:
        fig, ax = plt.subplots()
    for path, start_idx in zip(paths, start_ids):
        ax.plot([i + start_idx for i in range(len(path.phasepoints))],
                [ph[0] for ph in path.phasepoints], "-x", **kwargs)
        # plot the first and last point again to highlight start/end phasepoints
        # it must have the same color as the line for the path
        ax.plot(start_idx, path.phasepoints[0][0], "^",
                color=ax.lines[-1].get_color(), ms = 7)
        ax.plot(start_idx + len(path.phasepoints) - 1,
                path.phasepoints[-1][0], "v",
                color=ax.lines[-1].get_color(), ms = 7)
    if intfs is not None:
        for intf in intfs:
            ax.axhline(intf, color="k", ls="--", lw=.5)
    if ax is None:
        fig.show()
    

def overlay_paths(path, paths):
    """Searches for sequences of phasepoints that are identical in path and
    the list paths contained in paths. Returns a list containing a tuple for 
    each element of paths. 
    The first element of a tuple is the length of the largest sequence of 
    identical phasepoints, the second element is the index where the identical
    sequence starts in the path of paths.

    Parameters:
    -----------
    path: Path object
        Path to compare to. path.phasepoints are the phasepoints. A phasepoint
        is a tuple (x, v) of floats. We care only about the x coordinate. 
    paths: list of Path objects
        Paths to compare to path. Each path.phasepoints is a list of phasepoints
        (see above).

    Returns:
    --------
    list of tuples
        List of tuples (length of identical sequence, start index of identical
        sequence in path.phasepoints) for each path in paths.
    """
    result = []

    for p in paths:
        max_length = 0
        start_index = -1
        path_start_idx = -1  # Initialize the start index w.r.t. paths

        for i in range(len(p.phasepoints)):
            for j in range(len(path.phasepoints)):
                length = 0
                while i + length < len(p.phasepoints) and j + length < len(path.phasepoints) and \
                        p.phasepoints[i + length] == path.phasepoints[j + length]:
                    length += 1

                if length > max_length:
                    max_length = length
                    start_index = j
                    path_start_idx = i

        result.append((max_length, start_index, path_start_idx))

    return result

def remove_lines_from_file(fn, n=1):
    """Remove the last n lines from a file.
    
    Parameters
    ----------
    fn : str
        File name
    n : int, optional
        Number of lines to remove, by default 1

    """
    # We use: https://stackoverflow.com/questions/1877999/
    for i in range(n):
        with open(fn, "r+", encoding="utf-8") as f:
            # Move the pointer (similar to a cursor in a text editor) to the
            # end of the file
            f.seek(0, os.SEEK_END)
            # This code means the following code skips the very last character
            # in the file - i.e. in the case the last line is null we delete
            # the last line and the penultimate one
            pos = f.tell() - 1
            # Read each character in the file one at a time from the penultimate
            # character going backwards, searching for a newline character
            # If we find a new line, exit the search
            while pos > 0 and f.read(1) != "\n":
                pos -= 1
                f.seek(pos, os.SEEK_SET)
            # So long as we're not at the start of the file, delete all the
            # characters ahead of this position
            if pos > 0:
                f.seek(pos, os.SEEK_SET)
                f.truncate()
            # After truncating, we need to position the pointer at the end, 
            # on a new blank line..
            f.write("\n")