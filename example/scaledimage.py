"""
Simple intensity plot of matrix, similar to MATLAB imagesc()
"""
import numpy as NP
import matplotlib.pyplot as P
import matplotlib.ticker as MT
import matplotlib.cm as CM

def scaledimage(W, pixwidth=1, fig=None, grayscale=True):
    """
    Do intensity plot, similar to MATLAB imagesc()

    W = intensity matrix to visualize
    pixwidth = size of each W element
    fig = matplotlib figure to draw on 
    grayscale = use grayscale color map

    Rely on caller to .show()
    """
    # N = rows, M = column
    (N, M) = W.shape 
    # Need to create a new figure? 
    if(fig == None):
        fig = P.figure()
    # extents = Left Right Bottom Top
    exts = (0, pixwidth * M, 0, pixwidth * N)
    if(grayscale):
        fig.imshow(W,
                  interpolation='nearest',
                  cmap=CM.gray,
                  extent=exts)
    else:
        fig.imshow(W,
                  interpolation='nearest',
                  extent=exts)
    fig.xaxis.set_major_locator(MT.NullLocator())
    fig.yaxis.set_major_locator(MT.NullLocator())
