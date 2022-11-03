import numpy as np
from src.utils import get_adstock_matrix, compute_adstock_vector
from src.meBoot import MeBoot
from scipy.optimize import minimize
# added to the path


class BootModel:

    def __init__(self, data, features, out_name, boot_reps=99, intercept=True):
        self.data = data  # pandas
        self.features = features
        self.out_name = out_name
        self.intercept = intercept
        self.boot_reps = boot_reps
        self.samples = None

    def calc_decay(self, peak):
        length = len(self.data)
        # r^(l - p -1) = 0.001 => log(r)*(l-p-1) = log(0.001)
        return np.exp(np.log(0.001) / (length - peak - 1))

    def transform_vars(self, tr_dict):
        var_names = tr_dict.get('var_names')
        t_types = tr_dict.get('t_types')
        t_params = tr_dict.get('t_params')
        length = len(self.data)
        for i in range(len(var_names)):
            v_name = var_names[i]
            v_name_t = v_name + '_t'
            t_type = t_types[i]
            t_param = t_params[i]
            if 'adstock' in t_type:
                S = get_adstock_matrix(length, compute_adstock_vector(length, t_param))
                self.data[v_name_t] = np.matmul(S, self.data[v_name].values)
            else:
                pass
        return

    @staticmethod
    def fit_model(x, y, bounds=None):
        def objective(beta):
            return 0.5 * np.sum((np.matmul(x, beta) - y) ** 2), np.matmul(np.transpose(x), (np.matmul(x, beta) - y))

        def hessian(beta):
            return np.matmul(np.transpose(x), x)

        beta_init = np.zeros(shape=(x.shape[1],), dtype=np.float32)
        opt_res = minimize(objective, x0=beta_init, method='trust-constr', jac=True,
                           hess=hessian, bounds=bounds, options={'verbose': 0, 'xtol': 1e-10})

        return opt_res.x, opt_res.fun

    def set_samples(self):
        meb = MeBoot(reps=self.boot_reps)
        samples = []
        for feature in self.features:
            samples.append(meb.me_bootstrap(self.data.loc[:, feature]))

        self.samples = samples
        return

    def get_coef_dists(self):
        # get the coefficient dists using ME bootstrap
        if self.samples is None:
            self.set_samples()
        coef_list = []
        # initial model
        x = np.concatenate((np.ones(shape=(len(self.data), 1), dtype=np.float32), self.data[self.features].values),
                           axis=1)
        y = self.data[self.out_name].values
        initial_coefs = self.build_fit_model(x, y, [10, 11, 12, 13], 30) #self.fit_model(x, y)
        coef_list.append(initial_coefs)
        for i in range(self.boot_reps):
            # do the modeling
            x = np.ones(shape=(len(self.data), 1), dtype=np.float32)
            for j in range(len(self.features)):
                x = np.concatenate((x, self.samples[j].get('ensemble')[i, :].reshape(-1, 1)), axis=1)
            coef_list.append(self.build_fit_model(x, y, [10,11,12,13],30))
        return coef_list

    def build_fit_model(self, x, y, peak_range, ad_length):
        # first transform
        t_var_index = self.features.index('Volume') + 1
        best_score = np.inf
        best_coefs = np.zeros(shape=(x.shape[1],),dtype=np.float32)
        for p in peak_range:
            if ad_length > p + 1:
                r = np.exp(np.log(0.001)/(ad_length - p -1))
            else:
                break
            adst_mat = get_adstock_matrix(x.shape[0], compute_adstock_vector(x.shape[0], [ad_length, p, r]))
            x[:, t_var_index] = np.matmul(adst_mat, x[:,t_var_index])
            coefs, score = self.fit_model(x, y)
            if score < best_score:
                best_score = score
                best_coefs = coefs
                best_p = p

        return best_coefs
