### Code to solve 2+1 d quantum monte carlo XY model on square lattice 
### Jonathan Curtis
### 11/22/24

import numpy as np
from scipy import integrate as intg
import pickle 
import glauber


### Compatibility with demler_tools
def run_sims(save_filename,Lx,Ly,nsweeps,temps,replica,J_seed = None):

	### If a specific J matrix is not given we will by default generate an FM one  
	
	### We generate the J matrices and the nn coupling matrix 
	if J_seed is None:
		J_matrix = glauber.nn_coupling(-1.,Lx,Ly)
	else:
		J_matrix = glauber.nn_coupling_random(-1.,0.5,Lx,Ly,J_seed)
	
	nns = glauber.nn_indices(Lx,Ly)

	
	### Now we run the simulations for a number of sweeps for each temperature in the schedule, annealing as we go 
	### By default this will use random initial conditions for each replica 
	spins,times = glauber.anneal_dynamics(J_matrix,nns,nsweeps,temps)
	
	with open(save_filename, 'wb') as out_file:
        	pickle.dump((spins,J_matrix), out_file)
        	
