import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt 

def plot(Va, Vb, alpha, beta, save=False, path=None):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    ax[0].set_title("Volume plot.", fontsize=15)
    ax[0].plot(Va, linestyle='-', color='red', label='Best ask volume')
    ax[0].plot(Vb, linestyle='-', color='blue', label='Best bid volume')
    ax[0].set_xlabel("Time point", fontsize=15)
    ax[0].set_ylabel("Value", fontsize=15)
    ax[0].legend(loc='best')
    ax[0].grid()
    
    ax[1].set_title("Ask-bid plot.", fontsize=15)
    ax[1].plot(alpha, linestyle='-', color='red', label='Best ask')
    ax[1].plot(beta, linestyle='-', color='blue', label='Best bid')
    ax[1].set_xlabel("Time point", fontsize=15)
    ax[1].set_ylabel("Value", fontsize=15)
    ax[1].legend(loc='best')
    ax[1].grid()
    
    if save:
        plt.savefig(path)
    else:
        plt.show()
    
    
class Book:
    def __init__(self, params):
        self.Va0 = params['Va0']
        self.Vb0 = params['Vb0']
        self.alpha0 = params['alpha0']
        self.beta0 = params['beta0']

        self.mu = params['mu']
        self.nu = params['nu']
        self.kappa = params['kappa']
        self.delta = params['delta']
        
        self.d1 = params['d1']
        self.d2 = params['d2']
        self.d3 = params['d3']
        self.d4 = params['d4']
        
        self.T = params['T']
        self.N = params['N']
        
        self.sigma0 = params['sigma0']
        self.sigma1 = params['sigma1']
        self.sigma2 = params['sigma1']
        
    def simmulate(self):
        '''
        Function to generate sample path. 
        '''
        N = self.N
        T = self.T
        
        Va, Vb, alpha, beta = np.zeros(N + 1), np.zeros(N + 1), np.zeros(N + 1), np.zeros(N + 1)
        
        Xa, Xb, Ya, Yb = np.zeros(N + 1), np.zeros(N + 1), np.zeros(N + 1), np.zeros(N + 1)
        
        Va[0] = self.Va0
        Vb[0] = self.Vb0
        alpha[0] = self.alpha0
        beta[0] = self.beta0
        
        dt = T / N
        
        W = np.random.normal(0, 1, size=(5, N + 1))
        
        kappa = self.kappa
        d1, d2, d3, d4 = self.d1, self.d2, self.d3, self.d4
        sigma0, sigma1, sigma2 = self.sigma0, self.sigma1, self.sigma2
        nu, mu = self.nu, self.mu
        delta = self.delta
        
        
        for i in range(N):

            spread = alpha[i] - beta[i]
            imbalance = np.log(Vb[i] / Va[i])

            Xa[i + 1] = Xa[i] - Va[i] * 0.5 * (kappa * (mu - spread) * dt +
                                               (d1 + d2 * imbalance) * dt +
                                               sigma1 * np.sqrt(np.abs(spread)) * np.sqrt(dt) * (W[1, i + 1]) + 
                                               sigma0 * np.sqrt(dt)* (W[0, i + 1]))
            Xb[i + 1] = Xb[i] - Vb[i] * 0.5 * (kappa * (mu - spread) * dt +
                                               (d1 + d2 * imbalance) * dt +
                                               sigma1 * np.sqrt(np.abs(spread)) * np.sqrt(dt) * (W[2, i + 1]) + 
                                               sigma0 * np.sqrt(dt) * (W[0, i + 1]))

            Ya[i + 1] = Ya[i] - d3 * (nu - spread) * dt - delta * (mu - spread) * dt + sigma2 * np.sqrt(dt) * (W[3, i + 1] )

            Yb[i + 1] = Yb[i] + d4 * (nu - spread) * dt - delta * (mu - spread) * dt + sigma2 * np.sqrt(dt) * (W[4, i + 1] )

            Va[i + 1] = Va[i] + Va[i] * (Ya[i + 1] - Ya[i])

            Vb[i + 1] = Vb[i] + Vb[i] * (Yb[i + 1] - Yb[i])

            alpha[i + 1] = alpha[i] - (Xa[i + 1] - Xa[i]) / Va[i] + (Xa[i + 1] - Xa[i]) * (Ya[i + 1] - Ya[i]) / Va[i]

            beta[i + 1] = beta[i] - (Xb[i + 1] - Xb[i]) / Vb[i] + (Xb[i + 1] - Xb[i]) * (Yb[i + 1] - Yb[i]) / Vb[i]

        return alpha, beta, Va, Vb
            
    
    def modify_params(self, data):
        '''
        Function to approximate parameters by given data.
        '''
        K = 20
        t = data[:, 0]
        alpha = data[:, 1]
        beta = data[:, 2]
        Va = data[:, 3]
        Vb = data[:, 4]
        N = len(t)
        
        spread = alpha - beta
        mid = (alpha + beta) / 2.
        imb = np.log(Vb / Va)
        
        dt = np.diff(t)
        
        rv_s_total = (np.diff(spread) ** 2).sum()
        rv_m_total = (np.diff(mid) ** 2).sum()
        rv_i_total = (np.diff(imb) ** 2).sum()
        
        rv_s, rv_m, rv_i = np.zeros(K - 1), np.zeros(K - 1), np.zeros(K - 1)
        
        for i in range(K):
            index = range(i, N, k)
            rv_s[i] = (np.diff(spread[index]) ** 2).sum()
            rv_m[i] = (np.diff(mid[index]) ** 2).sum()
            rv_i[i] = (np.diff(imb[index]) ** 2).sum()
            
        rv_s = np.mean(rv_s - rv_s_total)
        rv_m = np.mean(rv_m - rv_m_total)
        rv_i = np.mean(rv_i - rv_i_total)
            
        sigma0 = np.sqrt((4. * rv_m - rv_s) / (T[-1] - T[0]))
        sigma2 = np.sqrt(2 * rv_i / (T[-1] - T[0]))
        int_s = (dt * spread[:-1]).sum()
        sigma1 = np.sqrt(rv_s * 2. / int_s)
        pass
        
        
        