import numpy as np
import time
import glob

import power_law_gaussian_no_U


# Get all files to test
files = glob.glob('../data/PF*.npy')


# For each file, run minimization
for f in files:
    filaments = np.load(f)

    start_time = time.time()
    solution = power_law_gaussian_no_U.run_optimize(filaments)
    print f
    print "Success:", solution.success
    print "Solution:", solution.x
    print "Elapsed time:", time.time()-start_time, "seconds"
    print ""
