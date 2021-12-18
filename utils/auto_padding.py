import math


def same_padding(k, s, d=1):
    """

    Args:
        k: kernel size
        s: stride
        d: dilation

    Returns: padding value

    """
    return math.ceil((d*(k-1)+1-s)/2)
