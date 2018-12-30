import numpy as np

"""
Implementation of the modified Kuhn-Munkres version of the article Lifted Wassertein Matcher for
Fast and Robust Topology Matching.

Inspired by http://csclab.murraystate.edu/~bob.pilgrim/445/munkres.html regarding the decomposition
in steps, and by https://github.com/bmc/munkres/blob/master/munkres.py.

"""


class ReducedMunkres:

    def __init__(self, costMat):
        """
        Args:
            costMat: (array of float) shape ((n+1), m)
        """
        self.n, self.m = costMat.shape
        self.costMat = costMat
        self.marked = np.full(costMat.shape, False, dtype=bool)
        self.starred = np.full(costMat.shape, False, dtype=bool)
        self.primed = np.full(costMat.shape, False, dtype=bool)
        self.row_covered = [False for i in range(self.n)]
        self.col_covered = [False for i in range(self.m)]
        self.independent = np.full(costMat.shape, False, dtype=bool)

        self.z1_r = None
        self.z1_c = None
        # Specific to the modified version of Munkres
        self.rowResiduals = None
        self.colResiduals = None
        assert self.m >= self.n - 1

    def _findUncovZero(self):
        """
        Find an uncovered zero in the matrix
        returns:
            z1: (tuple (row, col)) coordinates of a uncovered 0
        """
        uncov_zero = None
        for z1, value in np.ndenumerate(self.costMat):
            if value == 0 and not self.col_covered[z1[1]] and not self.row_covered[z1[0]]:
                uncov_zero = z1
        return uncov_zero

    def _findStarInRow(self, row):
        """
        Find the first starred element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        col = -1
        for i, star in enumerate(self.starred[row, :]):
            if star:
                col = i
                break
        return col

    def _findStarInCol(self, col):
        """
        Find the first starred element in the specified col. Returns
        the row index, or -1 if no starred element was found.
        """
        row = -1
        for i, star in enumerate(self.starred[:, col]):
            if star:
                row = i
                break
        return row

    def _findPrimeInRow(self, row):
        """
        Find the first primed element in the specified column. Returns
        the column index, or -1 if no primed element was found.
        """
        col = -1
        for i, primed in enumerate(self.primed[row, :]):
            if primed:
                col = i
                break
        return col

    def _isIndependent(self, pos):
        """
        Args:
            pos: (tuple (i,j)) the position of the tested zero for independence
        returns:
            ind: (bool) whether the zero is independent
        """
        ind = True
        row = pos[0]
        col = pos[1]
        if row == self.n - 1:
            pass
        else:
            # Other zero on the same row
            for j in range(self.m):
                if col != j and self.costMat[row, j] == 0:
                    ind = False
                    break
            # Other zero on the same column
            for i in range(self.n):
                if row != i and self.costMat[i, col] == 0:
                    ind = False
                    break
        return ind

    def star(self, pos):
        """
        Args:
            pos: (tuple (i,j)) the position of the tested zero 
        """
        canStar = True
        row = pos[0]
        col = pos[1]
        # Other zero starred on the same row
        for j in range(self.m):
            if col != j and self.starred[row, j]:
                canStar = False
                break
        # Other zero starred on the same column
        for i in range(self.n):
            if row != i and self.starred[i, col]:
                canStar = False
                break

        if canStar:
            self.starred[pos] = True

    def mark(self, pos):
        """
        Args:
            pos: (tuple (i,j)) the position of the zero to mark
        returns:
            nothing
        """
        self.marked[pos] = True

    def prime(self, pos):
        """
        Args:
            pos: (tuple (i,j)) the position of the zero to prime
        returns:
            nothing
        """
        self.primed[pos] = True

    def reset(self):
        """
        Reset all primes and all covers
        """
        self.row_covered = [False for i in range(self.n)]
        self.col_covered = [False for i in range(self.m)]
        self.primed = np.full(self.costMat.shape, False, dtype=bool)

    def step1(self):
        """
        Finds the smallest values of each column of the matrix, and subtract it from the columns
        """
        minValues = np.min(self.costMat, axis=0).reshape((1, -1))
        minValues = np.repeat(minValues, self.n, axis=0)
        self.costMat = self.costMat - minValues

        return 2

    def step2(self):
        """
        Stars the zeros of the matrix
        """

        for pos, value in np.ndenumerate(self.costMat):
            if value == 0:
                self.star(pos)

        return 3

    def step3(self):
        """
        Covers the column containing a starred zero
        """
        for _, col, starred in np.ndenumerate(self.starred):
            if starred and not self.col_covered[col]:
                self.col_covered[col] = True
            elif starred and self.col_covered[col]:
                raise ValueError("Column is already covered")
            else:
                pass

        return 4

    def step4(self):
        step = 0
        done = False

        while not done:
            # Find the first zero Z1 non-covered
            z1 = self._findUncovZero()

            # All the zeros are covered
            if z1 is None:
                done = True
                step = 6

            else:
                self.primed[z1[0], z1[1]] = True
                starCol = self._findStarInRow(z1[0])

                # if z1 is in the last row or there is no 0* in its row
                if starCol == -1:
                    self.z1_r = z1[0]
                    self.z1_c = z1[1]
                    done = True
                    step = 5  # augment path

                else:
                    col = list(self.starred[z1[0], :]).index(True)

                    # Cover the row, and uncover the column of Z1'
                    assert self.col_covered[col] == True
                    assert self.row_covered[z1[0]] == False
                    self.row_covered[z1[0]] = True
                    self.col_covered[col] = False

    def step5(self):
        """
        Augment path and reset covers and primes
        """

        # Reset covers and primes
        self.reset()

    def compute(self):
        pass
