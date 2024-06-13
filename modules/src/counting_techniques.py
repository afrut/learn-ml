# ----------------------------------------------------------------------
#
# Reference Material Montgomery & Runger: Applied statistics and Probability
# for Engineers 7ed
#
# ----------------------------------------------------------------------

import math
import subprocess as sp

import numpy as np


def prod(ls: list) -> int:
    return np.prod(ls)


def permRepeat(n: int, r: int = None) -> int:
    if r is not None:
        return n**r
    else:
        return n**n


def perm(n: int = None, r: int = None, ls: list = None) -> int:
    if r != None:
        return (int)(math.factorial(n) / math.factorial(n - r))
    elif ls != None:
        x = [math.factorial(y) for y in ls]
        return (int)(math.factorial(np.array(ls).sum()) / np.prod(x))
    else:
        return math.factorial(n)


def comb(n: int, r: int) -> int:
    ret = math.factorial(n) / (math.factorial(n - r) * math.factorial(r))
    return (int)(ret)


if __name__ == "__main__":
    sp.call("cls", shell=True)

    # ----------------------------------------------------------------------
    #
    # Multiplication Rule: The total number of possibilities is the product of
    # the number of ways each operation can be completed.
    #
    # An operation has 3 steps. Step 1 can be completed 7 different ways, step
    # 2 can be completed 2 different ways, and step 3 can be completed 5
    # different ways. The total number of ways the whole operation can be
    # completed is 7 x 2 x 5 = 70.
    #
    # ----------------------------------------------------------------------
    ls = [7, 2, 5]
    N = prod(ls)
    print("Total number of ways to complete an operation with the following steps:")
    cnt = 1
    msg = "    N ="
    for n in ls:
        print("    Step {0} can be completed {1} different ways".format(cnt, n))
        msg = msg + " {0} x".format(n)
    msg = msg[:-2]
    msg = msg + " = {0}".format(N)
    print(msg)
    print("\n\n")

    # ----------------------------------------------------------------------
    #
    # Permutations of a Set: The total number of different ways to arrange 5
    # elements is 5! = 5 x 4 x 3 x 2 x 1 = 120.
    #
    # ----------------------------------------------------------------------
    n = 5
    print(
        "The total number of different ways to arrange "
        + str(n)
        + " dfferent elements without repetition is:"
    )
    msg = "    N = "
    for x in range(n, 0, -1):
        msg = msg + " {0} x".format(x)
    N = perm(n)
    msg = msg[:-2] + " = {0}".format(N)
    print(msg)
    print(
        "The total number of different ways to arrange"
        + str(n)
        + " different elements WITH repetition is:"
    )
    msg = "    N = "
    for x in range(n, 0, -1):
        msg = msg + str(n) + " x "
    msg = msg[:-3] + " = " + str(permRepeat(n))
    print(msg)
    print("\n\n")

    # ----------------------------------------------------------------------
    #
    # Permutations of Subsets: The total number of different ways to arrange 3
    # elements taken from a set of 5 elements is 5!/(5 - 3)! = 5 * 4 * 3 = 60.
    #
    # ----------------------------------------------------------------------
    n = 5
    r = 3
    print(
        "The total number of different ways to arrange "
        + str(r)
        + " different elements taken\nfrom a set of "
        + str(n)
        + " elements is:"
    )
    print("    N = " + str(perm(n, r)))
    print("\n\n")

    # ----------------------------------------------------------------------
    #
    # Permutations of Similar Objects: The total number of different ways to
    # arrange 5 objects of type 1, 3 objects of type 2, and 4 objects of type 3
    # is (5 + 3 + 4)!/(5! x 3! x 4!)
    #
    # ----------------------------------------------------------------------
    ls = [3, 2, 4]
    print("The total number of different ways to arrange the following:")
    cnt = 1
    for n in ls:
        print("    {0} of type {1}".format(n, cnt))
        cnt = cnt + 1
    print("    N = " + str(perm(ls=ls)))
    print("\n\n")

    # ----------------------------------------------------------------------
    #
    # Combinations: The number of different sets of r elements that can be
    # formed from n distinct elements is n!/(n - r)!/r!. A set of r elements
    # arranged differently counts as the same set.
    #
    # ----------------------------------------------------------------------
    n = 7
    r = 4
    print("The number of {0}-element combinations of {1} elements is:".format(r, n))
    print("    N = " + str(comb(n, r)))
    print("\n\n")
