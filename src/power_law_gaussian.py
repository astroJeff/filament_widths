import sys
import numpy as np
from scipy.stats import norm, powerlaw
from scipy.integrate import quad
import pickle
import emcee
import time
import corner
import matplotlib.pyplot as plt


def ln_posterior(x, data):

    sigma, gamma, L, U = x

    if L<0.0 or L > 6.0:
        return -np.inf
    if U<L or U > 20.0:
        return -np.inf
    if sigma<0.0 or sigma>3.0:
        return -np.inf
    if gamma<0.01 or gamma>10.0:
        return -np.inf

    return ln_likelihood(x, data)


def get_integrand(x, y, gamma, sigma):
    return np.power(x, -(gamma+1.0)) * np.exp(-0.5*((y-x)/sigma)**2)


def ln_likelihood(x, data):

    sigma, gamma, L, U = x

    N = len(data)

    def get_integrand(x, y, gamma, sigma):
        return np.power(x, -(gamma+1.0)) * np.exp(-0.5*((y-x)/sigma)**2)



    integral = np.zeros(N)
    for i in np.arange(len(data)):
        args = data[i], gamma, sigma
        sol = quad(get_integrand, L, U, args=args)
        integral[i] = sol[0]

    ln_likelihood = - N/2.0 * np.log(2.0*np.pi) \
                    - N * np.log(sigma) \
                    + N * np.log(gamma) \
                    - N * np.log(np.power(L, -gamma) - np.power(U, -gamma)) \
                    + np.sum(np.log(integral))

    # print sigma, gamma, L, U, integral
    # return integral
    return ln_likelihood


def run_emcee(data, ndim=4, nwalkers=32, nburn=100, nrun=100):

    p0 = np.ones((nwalkers, ndim))
    p0[:,0] = norm.rvs(loc=2.0, scale=0.1, size=nwalkers) # sigma
    p0[:,1] = norm.rvs(loc=3.0, scale=0.1, size=nwalkers) # gamma
    p0[:,2] = norm.rvs(loc=1.0, scale=0.1, size=nwalkers) # L
    p0[:,3] = norm.rvs(loc=10.0, scale=0.1, size=nwalkers) # U

    # Define sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, args=[data])

    # Run burn-in
    pos,prob,state = sampler.run_mcmc(p0, N=nburn)

    # Full run
    sampler.reset()
    pos,prob,state = sampler.run_mcmc(pos, N=nrun)

    return sampler



# Load data
if len(sys.argv) != 2:
    print "Error, must include a filename"
    sys.exit()
filename = sys.argv[1]
filaments = np.load(filename)


# Run sampler
start = time.time()
sampler = run_emcee(filaments, nburn=250, nrun=250)
print "Elapsed time:", time.time() - start, "seconds"


# Save sampler
fileout = filename[:-4] + str("_sampler.data")
pickle.dump(sampler, open(fileout, "wb"))




# Create a few plots
corner.corner(sampler.flatchain)
fileout = filename[:-4] + str("_corner.pdf")
plt.savefig(fileout)



# Plot posterior samples
fig, ax = plt.subplots(1,2, figsize=(10,4))
ax[1].hist(filaments, histtype='step', color='k', bins=100, normed=True, linewidth=2.0)
x = np.linspace(0.001, 5.0, 1000)

N = len(sampler.flatchain)

# Plot samples from the posterior
for i in np.random.randint(0, high=N, size=10):
    sigma, gamma, L, U = sampler.flatchain[i]

    # Plot true distribution
    y = gamma * x**(-(gamma+1)) / (L**(-gamma) - U**(-gamma))
    y[x<L] = 0.0
    ax[0].plot(x, y, alpha=0.2, color='k')

    # Plot observed distribution
    integral = np.zeros(1000)
    coeff = 1.0/np.sqrt(2.0*np.pi)/sigma * gamma / (L**(-gamma) - U**(-gamma))
    for i in np.arange(1000):
        args = x[i], gamma, sigma
        sol = quad(get_integrand, L, U, args=args)
        integral[i] = sol[0]
    ax[1].plot(x, coeff*integral, color='k', alpha=0.2)


ax[0].set_xlim(0.0, 0.3)
ax[1].set_xlim(0.0, 0.3)

ax[0].set_ylim(0, 60)
ax[1].set_ylim(0, 20)

ax[0].set_yticklabels([])
ax[1].set_yticklabels([])


fileout = filename[:-4] + str("_samples.pdf")
plt.tight_layout()
plt.savefig(fileout)
#plt.show()


# Exit program
sys.exit()
