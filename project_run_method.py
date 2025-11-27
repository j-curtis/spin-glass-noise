### Code to solve 2+1 d quantum monte carlo XY model on square lattice 
### Jonathan Curtis
### 11/22/24

import numpy as np
from scipy import integrate as intg
import pickle 
import glauber


### Compatibility with demler_tools
def run_MC_sims(save_filename,L,nsweeps,FM,temps,replica):

	if FM:
		J = glauber.
	
	M = int(M)
	T = QMC.calc_temperature(dt,M)
	sim = QMC(Ej,Ec,T,L,M)
	sim.over_relax = True
	sim.local_field_draw = False #True 
	sim.shuffle_sites = True 
	sim.set_sampling(nburn,nsample,nstep)


	### Now we handle the case of a hot start 
	if hot_start_filename is not None:
		sim.hot_start(hot_start_filename)
		
	sim.burn()
	sim.sample()

	with open(save_filename, 'wb') as out_file:
        	pickle.dump((sim.action_samples,sim.OP_samples,sim.vort_samples,sim.theta_samples), out_file)
        	
        ### now some smaller suffixed files
        ### action samples 
	with open(save_filename+"_action_samples",'wb') as out_file:
		pickle.dump(sim.action_samples, out_file)
        
        ### OP samples 
	with open(save_filename+"_OP_samples",'wb') as out_file:
		pickle.dump(sim.OP_samples,out_file)




