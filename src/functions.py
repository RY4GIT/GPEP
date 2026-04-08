import math
import numpy as np

########################################################################################################################
# ludcmp, lubksb, and linearsolver

# Source: https://phyweb.physics.nus.edu.sg/~phywjs/CZ5101/ludcmp.py
# Similar to Fortran GMET solution

# "Numerical Recipes" ludcmp() C code on page 46 translated into Python
# I make it as simple as possible, disregard efficiency.
# Here a is a list of list, n is integer (size of the matrix)
# index is a list, and d is also a list of size 1
# Python list index starts from 0.  So matrix index is from 0 to n-1.


def ludcmp(a, n, indx, d):
    d[0] = 1.0
    # looking for the largest a in each row and store it in vv as inverse
    # We need a new list same size as indx, for this we use .copy()
    vv = indx.copy()
    for i in range(0, n):
        big = 0.0
        for j in range(0, n):
            temp = math.fabs(a[i][j])
            if temp > big:
                big = temp
        vv[i] = 1.0 / big
    #
    # run Crout's algorithm
    for j in range(0, n):
        # top half & bottom part are combined
        # but the upper limit l for k sum is different
        big = 0.0
        for i in range(0, n):
            if i < j:
                l = i
            else:
                l = j
            sum = a[i][j]
            for k in range(0, l):
                sum -= a[i][k] * a[k][j]
            # end for k
            a[i][j] = sum
            # for bottom half, we keep track which row is larger
            if i >= j:
                dum = vv[i] * math.fabs(sum)
                if dum >= big:
                    big = dum
                    imax = i
            # end if (i>= ...)
        # end for i
        # pivoting part, swap row j with row imax, a[j] is a whole row
        if j != imax:
            dum = a[imax]
            a[imax] = a[j]
            a[j] = dum
            d[0] = -d[0]
            vv[imax] = vv[j]
        # end if (j != ...)
        # divide by the beta diagonal value
        indx[j] = imax
        dum = 1.0 / a[j][j]
        for i in range(j + 1, n):
            a[i][j] *= dum
        # end for i
    # end for j


# end of def ludcmp


# We do backward substitution in lubksb() take the row swapped LU decomposed
# a, size n, and swapping indx, and b vector as input.  The output is
# in b after calling.
def lubksb(a, n, indx, b):
    ii = -1
    # forward
    for i in range(0, n):
        ip = indx[i]
        sum = b[ip]
        b[ip] = b[i]
        if ii != -1:
            for j in range(ii, i):
                sum -= a[i][j] * b[j]
        elif sum != 0:
            ii = i
        b[i] = sum
    # bote alpha_{ii} is 1 above
    #  backward
    for i in range(n - 1, -1, -1):
        sum = b[i]
        for j in range(i + 1, n):
            sum -= a[i][j] * b[j]
        b[i] = sum / a[i][i]


# end lubksb()


# unfortunately a is destroyed (become swapped LU)
def linearsolver(a, n, b):
    indx = list(range(n))
    d = [1]
    ludcmp(a, n, indx, d)
    x = b.copy()
    lubksb(a, n, indx, x)
    # print("x=", x)
    return x


########################################################################################################################
# basic regression utility functions


def least_squares_numpy(x, y, tx):
    # In fortran version, ludcmp and lubksb are used to calcualte matrix inversion
    # call ludcmp(a, indx, d)
    # call lubksb(a, indx, b)

    # In Python version, numpy is used to calculate matrix inversion
    c = np.matmul(tx, y)
    a = np.matmul(tx, x)

    n = np.shape(a)[0]
    b = np.zeros(n)

    deta = np.linalg.det(a)  # Compute the determinant of an array
    if deta == 0:
        # print('Singular matrix')
        b[:] = 0
    else:
        ainv = np.linalg.inv(a)
        b = np.matmul(ainv, c)

    return b


def least_squares_ludcmp(x, y, tx):
    # In fortran version, ludcmp and lubksb are used to calcualte matrix inversion
    # call ludcmp(a, indx, d)
    # call lubksb(a, indx, b)

    # In Python version, numpy is used to calculate matrix inversion
    c = np.matmul(tx, y)
    a = np.matmul(tx, x)
    n = np.shape(a)[0]
    b = linearsolver(list(a), n, list(c))

    return b


def logistic_regression(x, tx, yp):
    # nstn: station number
    # nvars: station attributes (1, lat/lon/...), 1 is for regression
    nstn, nvars = np.shape(x)

    b = np.zeros(nvars)  # regression coefficients (beta)
    p = np.zeros(nstn)  # estimated probability of occurrence
    f = 0  # flag: continue or stop loops
    it = 0  # iteration times

    while f != 1:
        # check for divergence
        xb = -np.matmul(x, b)
        if np.any(xb > 50):
            f = 1
        else:
            p = 1 / (1 + np.exp(xb))

        # check for divergence
        if np.any(p > 0.9999):
            # logistic regression diverging
            f = 1
        else:
            v = np.zeros([nstn, nstn])  # diagonal variance matrix
            for i in range(nstn):
                v[i, i] = p[i] * (1 - p[i])
            xv = np.matmul(v, x)
            yn = (
                yp - p
            )  # difference between station occurrence (0/1) and estimated probability: Bnew -Bold in Martyn and Slater 2006
            bn = least_squares_ludcmp(xv, yn, tx)

            # check: converging
            if np.any(np.abs(bn) > 1e-4):
                f = 0
            else:
                f = 1

            # check: iteration times
            if it > 20:
                f = 1
            else:
                f = 0

            b = b + bn  # update coefficients

        it = it + 1

    return b


# ### GMET original linear logistic regression
# def weight_linear_regression(nearinfo, weightnear, datanear, tarinfo):
#     # # nearinfo: predictors from neighboring stations
#     # # [station number, predictor number + 1] array with the first column being ones
#     # nearinfo = np.zeros([nnum, npred+1])
#     #
#     # # weightnear: weight of neighboring stations
#     # # [station number, station number] array with weights located in the diagonal
#     # weightnear = np.zeros([nnum, nnum])
#     # for i in range(nnum):
#     #     weightnear[i, i] = 123
#     #
#     # # tarinfo:  predictors from target stations
#     # # [predictor number + 1] vector with the first value being one
#     # tarinfo = np.zeros(npred+1)
#     #
#     # # datanear: data from neighboring stations. [station number] vector
#     # datanear = np.zeros(nnum)

#     # start regression
#     w_pcp_red = np.diag(np.squeeze(weightnear))
#     tx_red = np.transpose(nearinfo)
#     twx_red = np.matmul(tx_red, w_pcp_red)
#     b = least_squares_ludcmp(nearinfo, datanear, twx_red)
#     datatar = np.dot(tarinfo, b)

#     return datatar

# def weight_logistic_regression(nearinfo, weightnear, datanear, tarinfo):

#     try:
#         w_pcp_red = np.diag(np.squeeze(weightnear))

#         # if len(np.unique(datanear)) == 1:
#         #     poe = datanear[0] # all 0 (no rain) or all 1 (rain everywhere)
#         # else:
#         tx_red = np.transpose(nearinfo)
#         twx_red = np.matmul(tx_red, w_pcp_red)
#         b = logistic_regression(nearinfo, twx_red, datanear)
#         if np.all(b == 0) or np.any(np.isnan(b)):
#             poe = np.dot(weightnear, datanear)
#         else:
#             zb = - np.dot(tarinfo, b)
#             poe = 1 / (1 + np.exp(zb))

#     except:
#         poe = sklearn_weight_logistic_regression(nearinfo, weightnear, datanear, tarinfo)

#     return poe
# ### GMET original linear logistic regression


# ### sklearn linear and logistic regression
# def weight_logistic_regression(nearinfo, weightnear, datanear, tarinfo):

#     try:
#         poe = sklearn_weight_logistic_regression(nearinfo, weightnear, datanear, tarinfo)
#     except:
#         poe = np.nan

#     return poe

# def weight_linear_regression(nearinfo, weightnear, datanear, tarinfo):
#     model = LinearRegression()
#     model = model.fit(nearinfo, datanear, sample_weight=weightnear)
#     datatar = model.predict(tarinfo[np.newaxis, :])
#     return datatar
# ### sklearn linear and logistic regression
