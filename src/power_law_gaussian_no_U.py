import sys
import numpy as np
from numpy.random import multivariate_normal
from scipy.stats import norm, powerlaw, truncnorm
from scipy.integrate import quad
from scipy.optimize import minimize
import pickle
import emcee
import time
import corner
import matplotlib.pyplot as plt

upper = 20.0


def ln_posterior(x, data):

    sigma, gamma, L = x

    if L<0.0 or L > 0.5:
        return -np.inf
    if sigma<0.0 or sigma>3.0:
        return -np.inf
    if gamma<0.01 or gamma>10.0:
        return -np.inf

    lp = -np.log(sigma)

    return ln_likelihood(x, data) + lp


def ln_posterior_minimize(x, data):
    N = len(data)

    sigma, gamma, L = x

    if L<0.0 or L > 0.5:
        return np.inf
    if sigma<0.0 or sigma>3.0:
        return np.inf
    if gamma<0.01 or gamma>10.0:
        return np.inf

    ll = ln_likelihood(x, data)

    lp = -np.log(sigma)

    return -(ll + lp)




def get_integrand(x, y, gamma, sigma):
    return np.power(x, -(gamma+1.0)) * np.exp(-0.5*((y-x)/sigma)**2)


def ln_likelihood(x, data, method='MC'):

    N_samples = 1000

    sigma, gamma, L = x

    #N = len(data)
    N = len(data) / 100

    def get_integrand(x, y, gamma, sigma):
        return np.power(x, -(gamma+1.0)) * np.exp(-0.5*((y-x)/sigma)**2)

    integral = np.zeros(N)

    ran_indices = np.random.randint(len(data), size=N)
    # for i in np.arange(len(data)):


    if method == 'quad':
        for j in np.arange(N):
            i = ran_indices[j]
            args = data[i], gamma, sigma
            sol = quad(get_integrand, L, upper, args=args)
            integral[j] = sol[0]
    elif method == 'MC':
        ran_x = np.zeros(N)
        for i in np.arange(N):
            #a, b = (L - data[ran_indices[i]]) / sigma, (upper - data[ran_indices[i]]) / sigma
            #ran_x = truncnorm.rvs(a, b, loc=data[ran_indices[i]], scale=sigma, size=N_samples)

            ran_x = norm.rvs(loc=data[ran_indices[i]], scale=sigma, size=N_samples)
            #ids = np.intersect1d(np.where(ran_x>L)[0], np.where(ran_x<upper)[0])
            #integral[i] = (1.0/float(N_samples)) * np.sum(np.power(ran_x[ids],-(gamma+1.0)))
            integral[i] = (1.0/float(N_samples)) * np.sum(np.power(ran_x[ran_x>L],-(gamma+1.0)))

        #ran_x = multivariate_normal(mean=data[ran_indices], cov=sigma*np.identity(N), size=N_samples)
        #integral = (1.0/float(N_samples)) * np.sum(ran_x[ran_x>L]**(-gamma-1.0), axis=0)
    else:
        print "You must enter an appropriate method. Options: 'quad', 'MC'"
        sys.exit()



    ln_likelihood = - N/2.0 * np.log(2.0*np.pi) \
                    + N * np.log(gamma) \
                    - N * np.log(np.power(L, -gamma) - np.power(upper, -gamma)) \
                    + np.sum(np.log(integral))
#                    - N * np.log(sigma) \

    #print sigma, gamma, L, np.sum(np.log(integral)), N * np.log(sigma), ln_likelihood
    # return integral
    return ln_likelihood


def run_emcee(data, ndim=3, nwalkers=32, nburn=100, nrun=100):

    p0 = np.ones((nwalkers, ndim))
    p0[:,0] = norm.rvs(loc=0.1, scale=0.01, size=nwalkers) # sigma
    p0[:,1] = norm.rvs(loc=4.0, scale=0.1, size=nwalkers) # gamma
    p0[:,2] = norm.rvs(loc=0.05, scale=0.01, size=nwalkers) # L

    # Define sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, args=[data])

    # Run burn-in
    pos,prob,state = sampler.run_mcmc(p0, N=nburn)

    # Full run
    sampler.reset()
    pos,prob,state = sampler.run_mcmc(pos, N=nrun)

    return sampler



def run_optimize(data, ndim=3):

    # Initial Position
    p0 = np.array([0.1, 3.0, 0.01])

    def neg_likelihood(x, data):
        return -1.0 * ln_posterior(x, data)

    bounds=((0.001, 1.0), (0.01, 10.0), (0.0001, 1.0))

#    res = minimize(ln_posterior_minimize, p0, args=data, method='L-BFGS-B', bounds=bounds)
    res = minimize(ln_posterior_minimize, p0, args=data, method='Nelder-Mead', bounds=bounds)

    return res





# # Load data
# if len(sys.argv) != 2:
#     print "Error, must include a filename"
#     sys.exit()
# filename = sys.argv[1]
# filaments = np.load(filename)
#
#
# # Run optimizer
# print run_optimize(filaments)



#
#
# # Run sampler
# start = time.time()
# sampler = run_emcee(filaments, nburn=250, nrun=250)
# print "Elapsed time:", time.time() - start, "seconds"
#
#
# # Save sampler
# fileout = filename[:-4] + str("_sampler.data")
# pickle.dump(sampler, open(fileout, "wb"))
#
#
#
#
# # Create a few plots
# corner.corner(sampler.flatchain)
# fileout = filename[:-4] + str("_corner.pdf")
# plt.savefig(fileout)
#
#
#
# # Plot posterior samples
# fig, ax = plt.subplots(1,2, figsize=(10,4))
# ax[1].hist(filaments, histtype='step', color='k', bins=100, normed=True, linewidth=2.0)
# x = np.linspace(0.001, 5.0, 1000)
#
# N = len(sampler.flatchain)
#
# # Plot samples from the posterior
# for i in np.random.randint(0, high=N, size=10):
#     sigma, gamma, L, U = sampler.flatchain[i]
#
#     # Plot true distribution
#     y = gamma * x**(-(gamma+1)) / (L**(-gamma) - U**(-gamma))
#     y[x<L] = 0.0
#     ax[0].plot(x, y, alpha=0.2, color='k')
#
#     # Plot observed distribution
#     integral = np.zeros(1000)
#     coeff = 1.0/np.sqrt(2.0*np.pi)/sigma * gamma / (L**(-gamma) - U**(-gamma))
#     for i in np.arange(1000):
#         args = x[i], gamma, sigma
#         sol = quad(get_integrand, L, U, args=args)
#         integral[i] = sol[0]
#     ax[1].plot(x, coeff*integral, color='k', alpha=0.2)
#
#
# ax[0].set_xlim(0.0, 0.3)
# ax[1].set_xlim(0.0, 0.3)
#
# ax[0].set_ylim(0, 60)
# ax[1].set_ylim(0, 20)
#
# ax[0].set_yticklabels([])
# ax[1].set_yticklabels([])
#
#
# fileout = filename[:-4] + str("_samples.pdf")
# plt.tight_layout()
# plt.savefig(fileout)
# #plt.show()
#
#
# # Exit program
# sys.exit()
