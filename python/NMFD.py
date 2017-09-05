import warnings

import numpy as np

class NMFD(object):
    """
    """
    MU = 0
    EPS = np.spacing(1)
    DEBUG = True
    ITERFMT = 'Computing NMFD. iteration : {0:d}/{1:d}'

    def __init__(self, v, rank=3, template=10, iterations=50, w=None, h=None):
        """
        """
        # input spectrogram or scalogram
        self.v = v
        # data size
        self.m, self.n = self.v.shape
        # number of templates
        self.rank = rank
        # template size
        self.template = template
        # number or iterations
        self.iterations = iterations

        # bases
        if w is not None:
            self.w = w
            self.rank, self.template = self.w.shape[1:3]
        else:
            self.w = np.random.rand(self.m, self.rank, self.template)

        # weights
        if h is not None:
            self.h = h
            self.rank = self.h.shape[0]
        else:
            self.h = np.random.rand(self.rank, self.n)

        self.cost = np.zeros(self.iterations)

        # instance helper matrices
        self.one = np.ones((self.m, self.n))
        self.constraint = 1.02**np.arange(0, self.template) - 1.0
        self.constraint[0:2] = 0.0

    def compute_lambda(self):
        """
        computation of Lambda
        """
        if self._lambda is None:
            self._lambda = np.zeros((self.m, self.n))
        else:
            self._lambda[:] = 0.0
        for m in xrange(self.m):
            for r in xrange(self.rank):
                cv = np.convolve(self.w[m, r, :], self.h[r, :])
                self._lambda[m, :] += cv[0:self._lambda.shape[1]]

        self._vonlambda = self.v/(self._lambda + self.EPS)

    def cost_iter(self, i):
        """
        """
        self.cost[i] = (self.v*np.log(self.v/self._lambda) - self.v
                        + self._lambda).sum()

    def row_shift(self, a, t):
        shift_a = np.roll(a, -t).T
        if t > 0:
            shift_a[-t:, :] = 0.0
        return shift_a

    def avg_h(self):
        """
        average along t
        """
        hu = np.zeros((self.rank, self.n))
        hd = np.zeros((self.rank, self.n))
        for r in xrange(self.rank):
            for m in xrange(self.m):
                cv = np.convolve(self._vonlambda[m, :],
                                 np.flipud(self.w[m, r, :]))
                hu[r, :] += cv[self.template-1:self.template+self.n-1]
                cv = np.convolve(self.one[m, :], np.flipud(self.w[m, r, :]))
                hd[r, :] += cv[self.template-1:self.template+self.n-1]

        # average along t
        self.h *= hu/hd

    def update_wt(self):
        """
        update of Wt
        """
        for t in xrange(self.template):
            shift_h = self.row_shift(self.h, t)
            self.w[:, :, t] *= (np.dot(self._vonlambda, shift_h) /
                                (np.dot(self.one, shift_h) + self.EPS
                                 + self.MU * self.constraint[t]))

    def nmfd_iter(self, i):
        """
        nmfd iterator
        """
        # update of H for each value of t (which will be averaged)
        self.compute_lambda()
        self.cost_iter(i)
        self.avg_h()
        self.compute_lambda()
        self.update_wt()

    def nmfd(self):
        """
         NMFD(v, rank=3, template=10, iterations=50, w=None, h=None)
            NMFD as proposed by Smaragdis (Non-negative Matrix Factor
            Deconvolution; Extraction of Multiple Sound Sources from Monophonic
            Inputs). KL divergence minimization. The proposed algorithm was
            corrected.
        Input :
           - v : array_like
                magnitude spectrogram to factorize (is a MxN numpy array)
           - rank : int
                number of templates (unused if init_W or init_H is set)
           - template : int
                template size (in number of frames in the spectrogram) (unused
                if init_W is set)
           - iterations :
                number of iterations
           - w :
                initial value for W.
           - h :
                initial value for H.
        Output :
           - w : ndarray
                time/frequency template (MxRxT array, each template is TxM)
           - h : ndarry
                activities for each template (RxN array)

         Copyright (C) 2015 Romain Hennequin

         v : spectrogram MxN
         H : activation RxN
         Wt : spectral template MxR t = 0 to T-1
         W : MxRxT
        """

        for i in xrange(self.iterations):
            if self.DEBUG:
                warnings.warn(self.ITERFMT.format(i+1, self.iterations))
            self.nmfd_iter(i)

            return (self.w, self.h, self.cost)
