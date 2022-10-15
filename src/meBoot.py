# import random
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from scipy.stats import norm
from sklearn.preprocessing import scale
import datetime


# import multiprocessing as mp

# Imported from https://github.com/HandyGunawan/Medium/blob/master/MEBOOT/Complete_Code
# Made some fixes and modifications
# reps=999, trim=None, reachbound=True,
#                  expand_standard_deviation=True, force_central_limit=True, scl_adjustment=True, elaps=True
class MeBoot:
    def __init__(self, **kwargs):

        self.reps = kwargs.pop('reps', 999)
        self.trim = kwargs.pop('trim', None)
        self.reachbound = kwargs.pop('reachbound', True)
        self.expand_standard_deviation = kwargs.pop('expand_standard_deviation', True)
        self.force_central_limit = kwargs.pop('force_central_limit', True)
        self.scl_adjustment = kwargs.pop('scl_adjustment', True)
        self.elaps = kwargs.pop('elaps', True)

    def me_bootstrap(self, x):

        valid_input = False
        if self.trim is None:
            trim = {'trimval': 0.1, 'xmin': None, 'xmax': None}
        else:
            trim = self.trim

        if isinstance(x, pd.DataFrame):
            index = x.index
            x = x.to_numpy(dtype=object).T[0]
            valid_input = True
        elif isinstance(x, pd.Series):
            index = x.index
            x = x.to_numpy(dtype=object).T
            valid_input = True
        elif isinstance(x, np.ndarray):
            x = x
            valid_input = True
        else:
            print("only accept series, dataframe and arrays")

        if valid_input:
            current_time1 = datetime.datetime.now()
            n = len(x)
            xx = np.sort(x)
            order_x = np.argsort(x)
            z = np.array(pd.Series(xx).rolling(2).mean().dropna())
            dv = abs(np.diff(x))

            if trim['trimval'] is None:
                trimval = 0.1
            else:
                trimval = trim['trimval']

            dvtrim = scipy.stats.trim_mean(dv, trimval)

            if trim['xmin'] is None:
                xmin = xx[0] - dvtrim
            else:
                xmin = trim['xmin']

            if trim['xmax'] is None:
                xmax = xx[-1] + dvtrim
            else:
                xmax = trim['xmax']

            aux = pd.DataFrame([xx * 0.25, pd.Series(xx).shift(1) * 0.5,
                                pd.Series(xx).shift(2) * 0.25]).dropna(axis=1).sum(axis=0)
            desintxb = aux
            desintxb.loc[1] = 0.75 * xx[0] + 0.25 * xx[1];
            desintxb.loc[len(desintxb) + 1] = 0.25 * xx[-2] + 0.75 * xx[-1]
            desintxb.index = desintxb.index - 1;
            desintxb = np.array(desintxb.sort_index())  # shifting index
            #       desintxb is the desired mean

            ensemble = np.repeat(np.matrix(x), self.reps, axis=0)
            ensemble = np.array([self.shuffle_initial(n=ensemble.shape[1], z=z, xmin=xmin, xmax=xmax,
                                                      desintxb=desintxb) for p in ensemble])
            current_time2 = datetime.datetime.now()
            qseq = np.sort(ensemble)
            ensemble[:, order_x] = qseq

            if self.expand_standard_deviation:
                ensemble = self.expand_std(x=x, ensemble=ensemble, fiv=5)
                current_time3 = datetime.datetime.now()
            if self.force_central_limit:
                ensemble = self.force_clt(x=x, ensemble=ensemble)
                current_time4 = datetime.datetime.now()
            if self.scl_adjustment:
                zz = np.insert(z, 0, xmin);
                zz = np.insert(zz, -1, xmax)
                v = np.diff(zz ** 2) / 12
                xb = np.mean(x)
                s1 = np.sum((desintxb - xb) ** 2)
                uv = (s1 + np.sum(v)) / n
                desired_sd = np.std(x)
                actual_me_sd = np.sqrt(uv)
                out = desired_sd / actual_me_sd
                kappa = out - 1
                ensemble = ensemble + kappa * (ensemble - xb)
            else:
                kappa = None
            current_time5 = datetime.datetime.now()
            elapsr = [current_time2 - current_time1, current_time3 - current_time2, current_time4 - current_time3,
                      current_time5 - current_time4]

            if self.elaps: print("Elapsed Time:", elapsr)

            if not isinstance(x, np.ndarray):
                x = pd.DataFrame({'Values': x}, index=index)
                ensemble = pd.DataFrame(ensemble.T, index=index)
            return {'x': x, 'ensemble': ensemble, 'xx': xx, 'z': z, 'dv': dv,
                    'dvtrim': dvtrim, 'xmin': xmin, 'xmax': xmax, 'desintxb': desintxb,
                    'order_x': order_x, 'kappa': kappa, 'elaps': elapsr}

    def shuffle_initial(self, n, z, xmin, xmax, desintxb):
        p = np.random.uniform(size=n)
        q = np.full(n, -99999.0)
        for i in range(n - 2):
            if ((p > ((i + 1) / n)) & (p <= ((i + 2) / n))).any:
                ref23 = np.where((p > ((i + 1) / n)) & (p <= ((i + 2) / n)))[0]
                for j in range(len(ref23)):
                    k = ref23[j]
                    qq = z[i] + (z[i + 1] - z[i]) / (1 / n) * (p[k] - (i + 1) / n)
                    q[k] = qq + desintxb[i + 1] - 0.5 * (z[i] + z[i + 1])
        ref1 = np.where(p <= (1 / n))
        if len(ref1) > 0:
            qq = np.interp(p[ref1], [0, 1 / n], [xmin, z[1]])
            q[ref1] = qq
            if not self.reachbound:
                q[ref1] = qq + desintxb[0] - 0.5 * (z[0] + xmin)

        ref4 = np.where(p == ((n - 1) / n))
        if len(ref4) > 0:
            q[ref4] = z[-2]
        ref5 = np.where(p > ((n - 1) / n))
        if len(ref5) > 0:
            qq = np.interp(p[ref5], [(n - 1) / n, 1], [z[-2], xmax])
            q[ref5] = qq
            if not self.reachbound:
                q[ref5] = qq + desintxb[-1] - 0.5 * (z[-2] + xmax)

        return q

    @staticmethod
    def expand_std(x, ensemble, fiv=5):
        sdx = np.std(x, axis=0)
        sdf = np.insert(np.std(ensemble, axis=1), 0, sdx)
        sdfa = sdf / sdf[0]
        sdfd = sdf[0] / sdf
        mx = 1 + fiv / 100
        idd = np.where(sdfa < 1)
        if len(idd) > 0:
            sdfa[idd] = np.random.uniform(size=len(idd), low=1, high=mx)
        sdfdXsdfa = sdfd[1:] * sdfa[1:]
        idd = np.where(np.floor(sdfdXsdfa) > 0)
        if len(idd) > 0:
            ensemble[idd, :][0] = ensemble[idd, :][0].T.dot(np.diag(sdfdXsdfa[idd])).T
        return ensemble

    @staticmethod
    def force_clt(x, ensemble):
        bigj, n = ensemble.shape
        gm = np.mean(x)
        s = np.std(x, axis=0)
        smean = s / np.sqrt(bigj)
        xbar = np.mean(ensemble, axis=1)
        sortxbar = np.sort(xbar)
        oo = np.argsort(xbar)
        newbar = gm + norm.ppf(np.arange(1, bigj + 1) / (bigj + 1)) * smean
        scn = scale(newbar)
        newm = scn * smean + gm
        meanfix = newm - sortxbar
        out = ensemble
        out[oo, :] = ensemble[oo, :] + np.array([meanfix] * n).T
        return out
