"""
HAR-CAESar: Heterogeneous Autoregressive Conditional Autoregressive Expected Shortfall

This module implements the HAR-CAESar model, an extension of CAESar that incorporates
heterogeneous autoregressive dynamics based on the Heterogeneous Market Hypothesis
(Muller et al., 1997; Corsi, 2009).

The model includes daily, weekly (5-day), and monthly (22-day) return aggregates
to capture long-memory effects in tail risk.

Author: MSc Thesis Implementation
"""

import numpy as np
import multiprocessing as mp
from .caviar import CAViaR
import warnings


def compute_har_features(returns, weekly_window=5, monthly_window=22):
    """
    Compute HAR (Heterogeneous Autoregressive) features from a return series.
    
    INPUTS:
        - returns: ndarray
            Daily return series of length T.
        - weekly_window: int, optional
            Number of days for weekly average. Default is 5.
        - monthly_window: int, optional
            Number of days for monthly average. Default is 22.
    
    OUTPUTS:
        - har_features: dict with keys 'daily', 'weekly', 'monthly'
            Each value is an ndarray of length T containing the lagged features.
            - 'daily': r_{t-1} (simply lagged returns)
            - 'weekly': average of r_{t-1}, ..., r_{t-5}
            - 'monthly': average of r_{t-1}, ..., r_{t-22}
    
    Note: For the first few observations where full windows are not available,
    partial averages are computed using available data.
    """
    T = len(returns)
    
    # Daily: just lagged returns (r_{t-1} for forecasting r_t)
    daily = np.zeros(T)
    daily[1:] = returns[:-1]
    daily[0] = returns[0]  # Use first return as initial value
    
    # Weekly: rolling 5-day average of lagged returns
    weekly = np.zeros(T)
    for t in range(T):
        if t == 0:
            weekly[t] = returns[0]
        elif t < weekly_window:
            weekly[t] = np.mean(returns[:t])
        else:
            weekly[t] = np.mean(returns[t-weekly_window:t])
    
    # Monthly: rolling 22-day average of lagged returns
    monthly = np.zeros(T)
    for t in range(T):
        if t == 0:
            monthly[t] = returns[0]
        elif t < monthly_window:
            monthly[t] = np.mean(returns[:t])
        else:
            monthly[t] = np.mean(returns[t-monthly_window:t])
    
    return {'daily': daily, 'weekly': weekly, 'monthly': monthly}


class HAR_CAESar:
    """
    HAR-CAESar: Heterogeneous Autoregressive CAESar for joint VaR and ES estimation.
    
    This model extends the standard CAESar (Gatta et al., 2024) by incorporating
    HAR-style dynamics that capture long-memory effects in tail risk.
    
    The VaR equation is:
        q_t = β₀ + β₁^(d) r_{t-1}^+ + β₂^(d) r_{t-1}^- 
            + β₁^(w) r^(w)_{t-1}^+ + β₂^(w) r^(w)_{t-1}^-
            + β₁^(m) r^(m)_{t-1}^+ + β₂^(m) r^(m)_{t-1}^-
            + β_q q_{t-1} + β_e e_{t-1}
    
    The ES equation has the same structure with γ coefficients.
    
    Parameters per equation: 9 (intercept + 6 return coeffs + 2 AR terms)
    Total parameters: 18
    """
    
    def __init__(self, theta, lambdas=dict()):
        """
        Initialize the HAR-CAESar model.
        
        INPUTS:
            - theta: float
                Quantile level (e.g., 0.025 for 2.5% VaR).
            - lambdas: dict, optional
                Penalty weights for soft constraints.
                Keys: 'r' (ES residual), 'q' (VaR positivity), 'e' (monotonicity).
                Default is {'r': 10, 'q': 10, 'e': 10}.
        """
        self.theta = theta
        self.lambdas = {'r': 10, 'q': 10, 'e': 10}
        self.lambdas.update(lambdas)
        
        # HAR-CAESar specific parameters
        self.mdl_spec = 'HAR'
        self.n_parameters = 9  # Per equation: intercept + 6 return + 2 AR
        
        # Parameter indices for clarity
        # VaR equation (indices 0-8):
        #   0: intercept
        #   1: daily positive, 2: daily negative
        #   3: weekly positive, 4: weekly negative
        #   5: monthly positive, 6: monthly negative
        #   7: lagged q, 8: lagged e
        # ES equation (indices 9-17): same structure
        
        # Set the loop functions
        self.loop = self.R_HARloop
        self.joint_loop = self.Joint_HARloop
    
    def loss_function(self, v, r, y):
        """
        Compute the Barrera loss function for ES estimation (Step 2).
        
        INPUTS:
            - v: ndarray
                Value at Risk forecast.
            - r: ndarray
                Difference between ES forecast and VaR (r = e - q).
            - y: ndarray
                Target return series.
        
        OUTPUTS:
            - loss_val: float
                Barrera loss value.
        """
        loss_val = np.mean(
            (r - np.where(y < v, (y - v) / self.theta, 0)) ** 2
        ) + self.lambdas['r'] * np.mean(np.where(r > 0, r, 0))
        return loss_val
    
    def joint_loss(self, v, e, y):
        """
        Compute the Fissler-Ziegel (Patton) loss function for joint estimation (Step 3).
        
        INPUTS:
            - v: ndarray
                VaR forecast.
            - e: ndarray
                ES forecast.
            - y: ndarray
                Target return series.
        
        OUTPUTS:
            - loss_val: float
                Fissler-Ziegel loss value.
        """
        loss_val = np.mean(
            np.where(y <= v, (y - v) / (self.theta * e), 0) + v / e + np.log(-e)
        ) + self.lambdas['e'] * np.mean(np.where(e > v, e - v, 0)) + \
            self.lambdas['q'] * np.mean(np.where(v > 0, v, 0))
        return loss_val
    
    def ESloss(self, beta, y, q, r0, har_features):
        """
        Compute the Barrera loss for ES estimation.
        
        INPUTS:
            - beta: ndarray
                Parameters for ES equation. Shape is (9,).
            - y: ndarray
                Target return series.
            - q: ndarray
                VaR forecast from Step 1.
            - r0: float
                Initial value for r = e - q.
            - har_features: dict
                HAR features with keys 'daily', 'weekly', 'monthly'.
        
        OUTPUTS:
            - loss_val: float
                Loss value.
        """
        r = self.loop(beta, y, q, r0, har_features)
        loss_val = self.loss_function(q, r, y)
        return loss_val
    
    def Jointloss(self, beta, y, q0, e0, har_features):
        """
        Compute the Fissler-Ziegel loss for joint estimation.
        
        INPUTS:
            - beta: ndarray
                Parameters for both equations. Shape is (18,).
            - y: ndarray
                Target return series.
            - q0: float
                Initial VaR.
            - e0: float
                Initial ES.
            - har_features: dict
                HAR features.
        
        OUTPUTS:
            - loss_val: float
                Loss value.
        """
        q, e = self.joint_loop(beta, y, q0, e0, har_features)
        loss_val = self.joint_loss(q, e, y)
        return loss_val
    
    def R_HARloop(self, beta, y, q, r0, har_features, pred_mode=False):
        """
        Loop for ES residual (r = e - q) estimation with HAR specification.
        
        INPUTS:
            - beta: ndarray
                Parameters. Shape is (9,).
            - y: ndarray
                Target return series.
            - q: ndarray
                VaR forecast.
            - r0: float or list
                Initial value(s) for r.
            - har_features: dict
                HAR features with keys 'daily', 'weekly', 'monthly'.
            - pred_mode: bool, optional
                If True, r0 contains the last state for prediction.
        
        OUTPUTS:
            - r: ndarray
                ES residual (r = e - q).
        """
        rd = har_features['daily']
        rw = har_features['weekly']
        rm = har_features['monthly']
        
        # Precompute asymmetric coefficients
        rd_coeff = np.where(rd > 0, beta[1], beta[2])
        rw_coeff = np.where(rw > 0, beta[3], beta[4])
        rm_coeff = np.where(rm > 0, beta[5], beta[6])
        
        if pred_mode:
            # In prediction mode, r0 = [last_r]
            r = []
            r.append(beta[0] + 
                     rd_coeff[0] * rd[0] + 
                     rw_coeff[0] * rw[0] + 
                     rm_coeff[0] * rm[0] + 
                     beta[7] * r0[1] +  # lagged q from state
                     beta[8] * r0[2])   # lagged r from state
        else:
            r = [r0]
        
        # Main loop
        for t in range(1, len(y)):
            r_t = (beta[0] + 
                   rd_coeff[t] * rd[t] + 
                   rw_coeff[t] * rw[t] + 
                   rm_coeff[t] * rm[t] + 
                   beta[7] * q[t-1] + 
                   beta[8] * r[t-1])
            r.append(r_t)
        
        return np.array(r)
    
    def Joint_HARloop(self, beta, y, q0, e0, har_features, pred_mode=False):
        """
        Loop for joint VaR and ES estimation with HAR specification.
        
        INPUTS:
            - beta: ndarray
                Parameters for both equations. Shape is (18,).
                First 9 for VaR, next 9 for ES.
            - y: ndarray
                Target return series.
            - q0: float
                Initial VaR (not used in pred_mode).
            - e0: float or list
                Initial ES, or state [y_hist, q_hist, e_hist] in pred_mode.
            - har_features: dict
                HAR features.
            - pred_mode: bool, optional
                If True, e0 contains the last state for prediction.
        
        OUTPUTS:
            - q: ndarray
                VaR forecast.
            - e: ndarray
                ES forecast.
        """
        rd = har_features['daily']
        rw = har_features['weekly']
        rm = har_features['monthly']
        
        # Precompute asymmetric coefficients for VaR equation
        rd_coeff_q = np.where(rd > 0, beta[1], beta[2])
        rw_coeff_q = np.where(rw > 0, beta[3], beta[4])
        rm_coeff_q = np.where(rm > 0, beta[5], beta[6])
        
        # Precompute asymmetric coefficients for ES equation
        rd_coeff_e = np.where(rd > 0, beta[10], beta[11])
        rw_coeff_e = np.where(rw > 0, beta[12], beta[13])
        rm_coeff_e = np.where(rm > 0, beta[14], beta[15])
        
        if pred_mode:
            # In prediction mode, e0 = [y_history, q_history, e_history]
            q = []
            e = []
            # First prediction uses lagged values from state
            q.append(beta[0] + 
                     rd_coeff_q[0] * rd[0] + 
                     rw_coeff_q[0] * rw[0] + 
                     rm_coeff_q[0] * rm[0] + 
                     beta[7] * e0[1][-1] +   # lagged q from state
                     beta[8] * e0[2][-1])    # lagged e from state
            e.append(beta[9] + 
                     rd_coeff_e[0] * rd[0] + 
                     rw_coeff_e[0] * rw[0] + 
                     rm_coeff_e[0] * rm[0] + 
                     beta[16] * e0[1][-1] +  # lagged q from state
                     beta[17] * e0[2][-1])   # lagged e from state
        else:
            q = [q0]
            e = [e0]
        
        # Main loop
        for t in range(1, len(y)):
            q_t = (beta[0] + 
                   rd_coeff_q[t] * rd[t] + 
                   rw_coeff_q[t] * rw[t] + 
                   rm_coeff_q[t] * rm[t] + 
                   beta[7] * q[t-1] + 
                   beta[8] * e[t-1])
            e_t = (beta[9] + 
                   rd_coeff_e[t] * rd[t] + 
                   rw_coeff_e[t] * rw[t] + 
                   rm_coeff_e[t] * rm[t] + 
                   beta[16] * q[t-1] + 
                   beta[17] * e[t-1])
            q.append(q_t)
            e.append(e_t)
        
        return np.array(q), np.array(e)
    
    def optim4mp(self, yi, qi, r0, beta0, n_rep, har_features, pipend):
        """
        Optimization routine for multiprocessing (Step 2: ES estimation).
        
        INPUTS:
            - yi: ndarray
                Target return series.
            - qi: ndarray
                VaR forecast from Step 1.
            - r0: float
                Initial value for r.
            - beta0: ndarray
                Initial parameters.
            - n_rep: int
                Number of optimization repetitions.
            - har_features: dict
                HAR features.
            - pipend: multiprocessing.connection.Connection
                Pipe for communicating results.
        """
        from scipy.optimize import minimize
        
        # First iteration
        res = minimize(
            lambda x: self.ESloss(x, yi, qi, r0, har_features), beta0,
            method='SLSQP', options={'disp': False})
        beta_worker, fval_worker, exitflag_worker = res.x, res.fun, int(res.success)
        
        # Repeat until success or max iterations
        for _ in range(n_rep):
            res = minimize(
                lambda x: self.ESloss(x, yi, qi, r0, har_features), beta_worker,
                method='SLSQP', options={'disp': False})
            beta_worker, fval_worker, exitflag_worker = res.x, res.fun, int(res.success)
            if exitflag_worker == 1:
                break
        
        pipend.send((beta_worker, fval_worker, exitflag_worker))
    
    def joint_optim(self, yi, q0, e0, beta0, n_rep, har_features):
        """
        Joint optimization routine (Step 3).
        
        INPUTS:
            - yi: ndarray
                Target return series.
            - q0: float
                Initial VaR.
            - e0: float
                Initial ES.
            - beta0: ndarray
                Initial parameters. Shape is (18,).
            - n_rep: int
                Number of optimization repetitions.
            - har_features: dict
                HAR features.
        
        OUTPUTS:
            - beta_worker: ndarray
                Optimized parameters.
        """
        from scipy.optimize import minimize
        
        # First iteration
        res = minimize(
            lambda x: self.Jointloss(x, yi, q0, e0, har_features), beta0,
            method='SLSQP', options={'disp': False})
        beta_worker, exitflag_worker = res.x, int(res.success)
        
        # Repeat until success or max iterations
        for _ in range(n_rep):
            res = minimize(
                lambda x: self.Jointloss(x, yi, q0, e0, har_features), beta_worker,
                method='SLSQP', options={'disp': False})
            beta_worker, exitflag_worker = res.x, int(res.success)
            if exitflag_worker == 1:
                break
        
        return beta_worker
    
    def fit(self, yi, seed=None, return_train=False, q0=None, nV=102, n_init=3, n_rep=5):
        """
        Fit the HAR-CAESar model.
        
        INPUTS:
            - yi: ndarray
                Target return series.
            - seed: int or None, optional
                Random seed for reproducibility.
            - return_train: bool, optional
                If True, return fitted values. Default is False.
            - q0: list or None, optional
                [initial VaR, initial ES]. If None, computed from data.
            - nV: int, optional
                Number of random initializations. Default is 102.
            - n_init: int, optional
                Number of best initializations to use. Default is 3.
            - n_rep: int, optional
                Number of optimization repetitions. Default is 5.
        
        OUTPUTS:
            - dict with keys 'qi', 'ei', 'beta' (if return_train=True)
        """
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        
        if isinstance(yi, list):
            yi = np.array(yi)
        
        # Compute HAR features
        self.har_features_train = compute_har_features(yi)
        
        #-------------------- Step 0: CAViaR for initial VaR guess
        # Use standard CAViaR (AS specification) for initial VaR
        cav_res = CAViaR(self.theta, 'AS', p=1, u=1).fit(
            yi, seed=seed, return_train=True, q0=q0)
        qi, beta_cav = cav_res['qi'], cav_res['beta']
        del cav_res
        
        #-------------------- Step 1: Initialization
        if isinstance(q0, type(None)):
            n_emp = int(np.ceil(0.1 * len(yi)))
            if round(n_emp * self.theta) == 0:
                n_emp = len(yi)
            y_sort = np.sort(yi[:n_emp])
            quantile0 = int(round(n_emp * self.theta)) - 1
            if quantile0 < 0:
                quantile0 = 0
            e0 = np.mean(y_sort[:quantile0 + 1])
            q0_val = y_sort[quantile0]
        else:
            q0_val = q0[0]
            e0 = q0[1]
        
        #-------------------- Step 2: Initial guess for ES parameters
        np.random.seed(seed)
        nInitialVectors = [nV // 3, self.n_parameters]
        beta0 = [np.random.uniform(0, 1, nInitialVectors)]
        beta0.append(np.random.uniform(-1, 1, nInitialVectors))
        beta0.append(np.random.randn(*nInitialVectors))
        beta0 = np.concatenate(beta0, axis=0)
        
        # Evaluate initial guesses
        AEfval = np.empty(nV)
        for i in range(nV):
            AEfval[i] = self.ESloss(beta0[i, :], yi, qi, e0 - q0_val, 
                                    self.har_features_train)
        beta0 = beta0[AEfval.argsort()][0:n_init]
        
        #-------------------- Step 3: Optimization - Step I (Barrera loss)
        beta = np.empty((n_init, self.n_parameters))
        fval_beta = np.empty(n_init)
        exitflag = np.empty(n_init)
        
        # Multiprocessing
        workers = []
        for i in range(n_init):
            parent_pipend, child_pipend = mp.Pipe()
            worker = mp.Process(
                target=self.optim4mp,
                args=(yi, qi, e0 - q0_val, beta0[i, :], n_rep, 
                      self.har_features_train, child_pipend))
            workers.append([worker, parent_pipend])
            worker.start()
        
        # Gather results
        for i, worker_list in enumerate(workers):
            worker, parent_pipend = worker_list
            beta_worker, fval_worker, exitflag_worker = parent_pipend.recv()
            worker.join()
            beta[i, :] = beta_worker
            fval_beta[i] = fval_worker
            exitflag[i] = exitflag_worker
        
        ind_min = np.argmin(fval_beta)
        self.beta_es = beta[ind_min, :]
        
        #-------------------- Step 4: Optimization - Step II (Patton/FZ loss)
        # Build initial joint parameters from CAViaR and ES estimates
        # CAViaR AS has 5 parameters: [intercept, y_pos, y_neg, q_lag, e_lag placeholder]
        # We need to map to HAR structure
        
        # Initialize VaR parameters: use CAViaR for daily, zeros for weekly/monthly
        beta_q_init = np.zeros(self.n_parameters)
        beta_q_init[0] = beta_cav[0]      # intercept
        beta_q_init[1] = beta_cav[1]      # daily positive
        beta_q_init[2] = beta_cav[2]      # daily negative
        beta_q_init[3] = 0.0              # weekly positive (new)
        beta_q_init[4] = 0.0              # weekly negative (new)
        beta_q_init[5] = 0.0              # monthly positive (new)
        beta_q_init[6] = 0.0              # monthly negative (new)
        beta_q_init[7] = beta_cav[3]      # lagged q
        beta_q_init[8] = 0.0              # lagged e (cross term)
        
        # Initialize ES parameters from ES step, adjusted
        beta_e_init = np.zeros(self.n_parameters)
        beta_e_init[0] = self.beta_es[0] + beta_q_init[0]
        beta_e_init[1] = self.beta_es[1] + beta_q_init[1]
        beta_e_init[2] = self.beta_es[2] + beta_q_init[2]
        beta_e_init[3] = self.beta_es[3]
        beta_e_init[4] = self.beta_es[4]
        beta_e_init[5] = self.beta_es[5]
        beta_e_init[6] = self.beta_es[6]
        beta_e_init[7] = self.beta_es[7] + beta_q_init[7] - self.beta_es[8]
        beta_e_init[8] = self.beta_es[8]
        
        joint_beta = np.concatenate([beta_q_init, beta_e_init])
        
        # Joint optimization
        joint_beta_temp = self.joint_optim(yi, q0_val, e0, joint_beta, n_rep,
                                           self.har_features_train)
        if not np.isnan(joint_beta_temp).any():
            joint_beta = joint_beta_temp
        
        self.beta = joint_beta
        
        # Compute fitted values
        qi, ei = self.joint_loop(self.beta, yi, q0_val, e0, self.har_features_train)
        
        # Store state for prediction (need history for HAR features)
        self.last_state = [yi[-22:], qi[-1:], ei[-1:]]  # Keep 22 for monthly avg
        self.last_y_full = yi  # Keep full history for HAR computation
        
        if return_train:
            return {'qi': qi, 'ei': ei, 
                    'beta': self.beta.reshape((2, self.n_parameters))}
    
    def predict(self, yf=np.array([])):
        """
        Predict VaR and ES for new observations.
        
        INPUTS:
            - yf: ndarray, optional
                New return observations. Default is empty array.
        
        OUTPUTS:
            - dict with keys 'qf', 'ef'
                VaR and ES forecasts.
        """
        if len(yf) == 0:
            return {'qf': np.array([]), 'ef': np.array([])}
        
        # Compute HAR features for prediction
        # Combine historical and new data for proper HAR computation
        y_combined = np.concatenate([self.last_y_full, yf])
        har_features_full = compute_har_features(y_combined)
        
        # Extract features for the forecast period
        n_hist = len(self.last_y_full)
        har_features_pred = {
            'daily': har_features_full['daily'][n_hist:],
            'weekly': har_features_full['weekly'][n_hist:],
            'monthly': har_features_full['monthly'][n_hist:]
        }
        
        # Run prediction loop
        qf, ef = self.joint_loop(self.beta, yf, None, self.last_state, 
                                  har_features_pred, pred_mode=True)
        
        # Update state
        if len(yf) > 0:
            self.last_y_full = np.concatenate([self.last_y_full, yf])
            # Keep last 22 for state (needed for HAR)
            if len(self.last_y_full) > 1000:  # Trim to avoid memory issues
                self.last_y_full = self.last_y_full[-500:]
            self.last_state = [
                np.concatenate([self.last_state[0], yf])[-22:],
                np.array([qf[-1]]),
                np.array([ef[-1]])
            ]
        
        return {'qf': qf, 'ef': ef}
    
    def fit_predict(self, y, ti, seed=None, return_train=True, q0=None, 
                    nV=102, n_init=3, n_rep=5):
        """
        Fit and predict in one call.
        
        INPUTS:
            - y: ndarray
                Full return series.
            - ti: int
                Training set length.
            - seed: int or None
                Random seed.
            - return_train: bool
                Return training predictions.
            - q0: list or None
                Initial [VaR, ES].
            - nV, n_init, n_rep: int
                Optimization parameters.
        
        OUTPUTS:
            - dict with keys 'qi', 'ei', 'qf', 'ef', 'beta'
        """
        yi, yf = y[:ti], y[ti:]
        
        if return_train:
            res_train = self.fit(yi, seed=seed, return_train=True, q0=q0,
                                nV=nV, n_init=n_init, n_rep=n_rep)
            res_test = self.predict(yf)
            return {
                'qi': res_train['qi'],
                'ei': res_train['ei'],
                'qf': res_test['qf'],
                'ef': res_test['ef'],
                'beta': self.beta.reshape((2, self.n_parameters))
            }
        else:
            self.fit(yi, seed=seed, return_train=False, q0=q0,
                    nV=nV, n_init=n_init, n_rep=n_rep)
            res_test = self.predict(yf)
            return {
                'qf': res_test['qf'],
                'ef': res_test['ef'],
                'beta': self.beta.reshape((2, self.n_parameters))
            }


# Convenience function for model selection
def HAR_CAESar_model(theta, lambdas=dict()):
    """
    Factory function for HAR-CAESar model.
    
    INPUTS:
        - theta: float
            Quantile level.
        - lambdas: dict, optional
            Penalty weights.
    
    OUTPUTS:
        - HAR_CAESar model instance.
    """
    return HAR_CAESar(theta, lambdas=lambdas)
