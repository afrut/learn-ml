# naive implementation of multilinear interpolation
import numpy as np
from IPython import embed


# limit-check the input vector against the table
def limitcheck(x, table):
    x = x.copy()
    xmin = table.min(axis=0)[0 : len(x)]
    idx = x < xmin
    x[idx] = xmin[idx]
    xmax = table.max(axis=0)[0 : len(x)]
    idx = x > xmax
    x[idx] = xmax[idx]
    return x


# compute the change in x
def computexd(x, x0, x1):
    if x1 - x0 == 0:
        return 0
    else:
        return (x - x0) / (x1 - x0)


# helper function for recursively multi-dimensionally indexing an array
def idxmultidim(idx, array):
    D = len(idx)
    ref = array
    idxref = 0
    while idxref <= D - 1:
        ref = ref[int(idx[idxref])]
        idxref = idxref + 1
    return ref


def interp(xd, y0, y1):
    return (y0 * (1 - xd)) + (y1 * xd)


def interpmd(x, table):
    D = len(x)
    assert D == table.shape[1] - 1

    # ----------------------------------------
    # find the 2**D points surrounding x
    # ----------------------------------------
    # container for points surrounding the point of interest
    surroundingpoints = np.empty([2 for idx in range(D)], dtype=object)
    yvals = np.empty([2 for idx in range(D)], dtype=object)

    # loop through all possible hi-lo combinations in D-dimensional space
    comb = np.zeros(D, dtype=np.int64)
    for idx in range(2**D):
        search = table
        for idxcomb in range(D):
            if comb[idxcomb] == 0:
                search = search[
                    search[:, idxcomb]
                    == search[search[:, idxcomb] <= x[idxcomb], idxcomb].max(),
                    :,
                ]
            else:
                search = search[
                    search[:, idxcomb]
                    == search[search[:, idxcomb] >= x[idxcomb], idxcomb].min(),
                    :,
                ]

        # store the search result
        ref = surroundingpoints
        refy = yvals
        idxref = 0
        while idxref < D - 2:
            ref = ref[comb[idxref]]
            refy = refy[comb[idxref]]
            idxref = idxref + 1
        ref[comb[idxref], comb[idxref + 1]] = np.ravel(search)
        refy[comb[idxref], comb[idxref + 1]] = np.ravel(search)[D]

        # find the first 0 index in c
        idxcomb = D - 1
        while comb[idxcomb] > 0 and idxcomb >= 0:
            comb[idxcomb] = 0  # set to 0
            idxcomb = idxcomb - 1  # move to the next element

        # account for overflow
        if idxcomb < 0:
            idxcomb = 0

        # add one
        comb[idxcomb] = 1

    # ----------------------------------------
    # interpolate and remove an axis
    # ----------------------------------------
    idxdimremove = 0
    while idxdimremove < D:
        # new remaining dimensions
        newD = D - idxdimremove - 1

        # compute xlo
        idx = np.array([0 for idxD in range(D)])
        xlo = idxmultidim(idx, surroundingpoints)[idxdimremove]

        # compute xhi
        idx = list()
        for idxD in range(D):
            if idxD == idxdimremove:
                idx.append(1)
            else:
                idx.append(0)
        xhi = idxmultidim(idx, surroundingpoints)[idxdimremove]

        # compute xd
        xd = computexd(x[idxdimremove], xlo, xhi)

        # loop through all possible hi-lo combinations in newD-dimensional space
        comb = np.zeros(newD, dtype=np.int64)
        newyvals = np.empty([2 for idx in range(newD)], dtype=object)
        for idx in range(2 ** (newD)):
            # compute ylo
            idxylo = np.hstack([np.array([0]), comb])
            ylo = idxmultidim(idxylo, yvals)

            # compute yhi
            idxyhi = np.hstack([np.array([1]), comb])
            yhi = idxmultidim(idxyhi, yvals)

            # populate new yvals
            ref = newyvals
            if newD > 2:
                idxref = 0
                while idxref < newD - 2:
                    ref = ref[comb[idxref]]
                    idxref = idxref + 1
                ref[comb[idxref], comb[idxref + 1]] = interp(xd, ylo, yhi)
            elif newD == 2:
                ref[comb[0], comb[1]] = interp(xd, ylo, yhi)
            elif newD == 1:
                ref[comb[0]] = interp(xd, ylo, yhi)
            else:
                ref = interp(xd, ylo, yhi)
                return ref

            # find the first 0 index in c
            if newD > 0:
                idxcomb = newD - 1
                while comb[idxcomb] > 0 and idxcomb >= 0:
                    comb[idxcomb] = 0  # set to 0
                    idxcomb = idxcomb - 1  # move to the next element

                # account for overflow
                if idxcomb < 0:
                    idxcomb = 0

                # add one
                comb[idxcomb] = 1

        # update yvals
        yvals = newyvals

        # remove increment the dimension to remove
        idxdimremove = idxdimremove + 1

    return None


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # generate some data
    x1 = np.arange(25)[:, np.newaxis]
    x2 = np.linspace(0, 223, 25)[:, np.newaxis]
    x3 = np.linspace(30, 150, 25)[:, np.newaxis]
    x = list()
    for v1 in x1:
        for v2 in x2:
            for v3 in x3:
                x.append((v1, v2, v3))
    x = np.squeeze(np.array(x), axis=2)
    y = (np.sqrt(x[:, 0]) + (0.7 * x[:, 1]) + x[:, 2] - 17)[:, np.newaxis]
    table = np.hstack([x, y])

    # sample coordinate of interest
    ox = np.array([20.34, 42.22, 66.4])
    x = limitcheck(ox, table)

    # ----------------------------------------
    # Find the 2^3 points surrounding the point of interest
    # c[0][1][0] is a point where the value of the second coordinate is greater
    # than the 2nd coordinate of the point of interest
    # ----------------------------------------
    c = np.empty((2, 2, 2), dtype=object)
    search = table[table[:, 0] == table[table[:, 0] <= x[0], 0].max(), :]
    search = search[search[:, 1] == search[search[:, 1] <= x[1], 1].max(), :]
    search = search[search[:, 2] == search[search[:, 2] <= x[2], 2].max(), :]
    c[0][0][0] = np.ravel(search)
    # print(c[0][0][0])

    search = table[table[:, 0] == table[table[:, 0] >= x[0], 0].min(), :]
    search = search[search[:, 1] == search[search[:, 1] <= x[1], 1].max(), :]
    search = search[search[:, 2] == search[search[:, 2] <= x[2], 2].max(), :]
    c[1][0][0] = np.ravel(search)
    # print(c[1][0][0])

    search = table[table[:, 0] == table[table[:, 0] <= x[0], 0].max(), :]
    search = search[search[:, 1] == search[search[:, 1] >= x[1], 1].min(), :]
    search = search[search[:, 2] == search[search[:, 2] <= x[2], 2].max(), :]
    c[0][1][0] = np.ravel(search)
    # print(c[0][1][0])

    search = table[table[:, 0] == table[table[:, 0] >= x[0], 0].min(), :]
    search = search[search[:, 1] == search[search[:, 1] >= x[1], 1].min(), :]
    search = search[search[:, 2] == search[search[:, 2] <= x[2], 2].max(), :]
    c[1][1][0] = np.ravel(search)
    # print(c[1][1][0])

    search = table[table[:, 0] == table[table[:, 0] <= x[0], 0].max(), :]
    search = search[search[:, 1] == search[search[:, 1] <= x[1], 1].max(), :]
    search = search[search[:, 2] == search[search[:, 2] >= x[2], 2].min(), :]
    c[0][0][1] = np.ravel(search)
    # print(c[0][0][1])

    search = table[table[:, 0] == table[table[:, 0] >= x[0], 0].min(), :]
    search = search[search[:, 1] == search[search[:, 1] <= x[1], 1].max(), :]
    search = search[search[:, 2] == search[search[:, 2] >= x[2], 2].min(), :]
    c[1][0][1] = np.ravel(search)
    # print(c[1][0][1])

    search = table[table[:, 0] == table[table[:, 0] <= x[0], 0].max(), :]
    search = search[search[:, 1] == search[search[:, 1] >= x[1], 1].min(), :]
    search = search[search[:, 2] == search[search[:, 2] >= x[2], 2].min(), :]
    c[0][1][1] = np.ravel(search)
    # print(c[0][1][1])

    search = table[table[:, 0] == table[table[:, 0] >= x[0], 0].min(), :]
    search = search[search[:, 1] == search[search[:, 1] >= x[1], 1].min(), :]
    search = search[search[:, 2] == search[search[:, 2] >= x[2], 2].min(), :]
    c[1][1][1] = np.ravel(search)
    # print(c[1][1][1])

    # ----------------------------------------
    # Interpolate along the first dimension
    # ----------------------------------------
    c1 = np.empty((2, 2), dtype=object)
    xd = computexd(x[0], c[0][0][0][0], c[1][0][0][0])
    c1[0][0] = interp(xd, c[0][0][0][3], c[1][0][0][3])
    c1[0][1] = interp(xd, c[0][0][1][3], c[1][0][1][3])
    c1[1][0] = interp(xd, c[0][1][0][3], c[1][1][0][3])
    c1[1][1] = interp(xd, c[0][1][1][3], c[1][1][1][3])

    # ----------------------------------------
    # Interpolate along the second dimension
    # ----------------------------------------
    c2 = np.empty(2, dtype=object)
    xd = computexd(x[1], c[0][0][0][1], c[0][1][0][1])
    c2[0] = interp(xd, c1[0][0], c1[1][0])
    c2[1] = interp(xd, c1[0][1], c1[1][1])

    # ----------------------------------------
    # Interpolate along the third dimension
    # ----------------------------------------
    c3 = None
    xd = computexd(x[2], c[0][0][0][2], c[0][0][1][2])
    c3 = interp(xd, c2[0], c2[1])

    # multi-linear interpolation
    yval = interpmd(x, table)
    print(table)
    print(x)
    print(c)
    print(yval)

    assert c3 == yval, "Incorrect interpolated outputs {0:.4f} vs {1:.4f}".format(
        c3, yval
    )
