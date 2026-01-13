### Jonathan Curtis
### 11/22/24

import numpy as np
from scipy import integrate as intg
import pickle 
import glauber
import noise_methods as nm 
from demler_tools.file_manager import path_management, file_management, io


### Demler tools simulation method which saves the entire spin trajectory (costly in terms of storage)
def run_sims_full_trajectory(save_filename,Lx,Ly,nsweeps,temps,replica,J_seed = None,start_polarized = False):
	J_seed = int(J_seed) 
	replica = int(replica) 

	### If a specific J matrix is not given we will by default generate an FM one  
	
	### We generate the J matrices and the nn coupling matrix 
	if J_seed is None:
		J_matrix = glauber.nn_coupling(-1.,Lx,Ly)
	else:
		J_matrix = glauber.nn_coupling_random(-1.,0.5,Lx,Ly,J_seed)
	
	nns = glauber.nn_indices(Lx,Ly)

	
	### Now we run the simulations for a number of sweeps for each temperature in the schedule, annealing as we go 
	### By default this will use random initial conditions for each replica 
	if start_polarized:
		polarized = glauber.initialize_spins(Lx,Ly)
		spins,times,energies = glauber.anneal_dynamics(J_matrix,nns,nsweeps,temps,initial_spins=polarized)
		
	else:
		spins,times,energies = glauber.anneal_dynamics(J_matrix,nns,nsweeps,temps)
		
	with open(save_filename, 'wb') as out_file:
        	pickle.dump((spins,energies,J_matrix), out_file) ### We store the output spin trajectory, the annealing schedule, and the J config
        	
### Saves more compact derived observables only   	
def run_sims(save_filename,Lx,Ly,nsweeps,temps,distances,replica,J_seed = None,start_polarized = False):
	J_seed = int(J_seed) 
	replica = int(replica) 

	### If a specific J matrix is not given we will by default generate an FM one  
	
	### We generate the J matrices and the nn coupling matrix 
	if J_seed is None:
		J_matrix = glauber.nn_coupling(-1.,Lx,Ly)
	else:
		J_matrix = glauber.nn_coupling_random(-1.,0.5,Lx,Ly,J_seed)
	
	nns = glauber.nn_indices(Lx,Ly)

	
	### Now we run the simulations for a number of sweeps for each temperature in the schedule, annealing as we go 
	### By default this will use random initial conditions for each replica 
	if start_polarized:
		polarized = glauber.initialize_spins(Lx,Ly)
		spins,times,energies = glauber.anneal_dynamics(J_matrix,nns,nsweeps,temps,initial_spins=polarized)
		
	else:
		spins,times,energies = glauber.anneal_dynamics(J_matrix,nns,nsweeps,temps)
		
	### Due to large memory of spin configurations we will now compute derived observables to save 
	### 1) Energy vs time 
	### 2) Magnetization vs time 
	### 3) Edwards-Anderson OP vs time 
	### 4) Local noise for different distances vs time 
	
	magnetization = glauber.calc_mag(spins) ### Extract magnetization dynamics 
	
	sample_step = 30 ### Sample every ___ steps 
	chop = int(nsweeps//5) ### Chop off first __% of each trajectory 
	qea = glauber.calc_ea(spins,sample_step,chop) 
	
	noise = nm.calc_local_noise(spins,distances,Lx,Ly) 
	
	with open(save_filename, 'wb') as out_file:
        	pickle.dump((J_matrix,energies,magnetization,qea,noise), out_file) ### We store the output spin trajectory, the annealing schedule, and the J config
        	
        	
        	
### Processing scripts for quench and annealing runs 
### Recover and process jobs for quench dynamics 
### Recovers also only a single seed 
def process_quench(timestamp,get_seed=0,get_replicas=None):
	print("Recovering quench calculation.")
	### We expect that for quench each run will be one temperature and possibly a few replicas of this 

	### First lets extract all of the different J matrices 
	job_no = io.recover_job_no(timestamp = timestamp)
	print(f"Total jobs: {job_no}")

	seeds = [] ### We will also use a list to store the different seeds
	jobs_by_seed = {} ### Jobs which have the given seed 
	### Get all the different seeds and replicas for each job with a particular seed
	for job in range(job_no):
		inputs,data = io.get_results(timestamp=timestamp,run_index=job)
		seed = str(int(inputs['J_seed']))

		if seed not in seeds: 
			seeds.append(seed) 
			jobs_by_seed[seed] = [ job ]

		else:
			(jobs_by_seed[seed]).append(job)

	### Now we process all for a given seed   
	jobs = jobs_by_seed[seeds[get_seed]] ### List of jobs for requested seed
	temps = [] 
	replicas = [] 
	jobs_by_temp = {} ### Dictionary with jobs for each temperature
	jobs_by_replica = {} ### Dictionary with jobs for each replica 
	spins_by_job = {} ### Store entries with key given by job number 
	energies_by_job = {} ### Store energies with key given by job number 

	### Only loop over jobs with the correct seed 
	for job in jobs:
		inputs, data = io.get_results(timestamp = timestamp,run_index = job)
		try:
			spins, energies,J = data
		except:
			spins, J = data
			energies = np.array([ None ])
			print("Legacy dataset, setting energies to None")
			
		replica = int(inputs['replica'])
		temp = inputs['temps']

		if temp not in temps: temps.append(temp)
		if replica not in replicas: replicas.append(replica) 

		print(f"Job: {job}, replica: {replica}, temp: {temp}, seed: {seed}")

		spins_by_job[job] = spins[0,...]
		energies_by_job[job] = energies[0,...]

		if temp not in jobs_by_temp.keys():
			jobs_by_temp[temp] = [ job ]
		else:
		    	jobs_by_temp[temp].append(job) 
		
		if get_replicas is None or replica in get_replicas:
			if replica not in jobs_by_replica.keys():
		    		jobs_by_replica[replica] = [ job ]

			else:
		    		jobs_by_replica[replica].append(job) 
		    		
		else:
			continue
    
		spins_stacked_replica = []
		energies_stacked_replica = [] 
		for replica in replicas:
			spins_stacked_temp = []
			energies_stacked_temp = []
			for temp in temps:
				for job in jobs_by_temp[temp]:
					if job in jobs_by_replica[replica]: 
						spins_stacked_temp.append(spins_by_job[job])
						energies_stacked_temp.append(energies_by_job[job])
		    
			spins_stacked_temp = np.stack(spins_stacked_temp) 
			energies_stacked_temp = np.stack(energies_stacked_temp)

			spins_stacked_replica.append(spins_stacked_temp) 
			energies_stacked_replica.append(energies_stacked_temp)

	spins = np.stack(spins_stacked_replica) 
	energies = np.stack(energies_stacked_replica)
	temps = np.array(temps) 
	    
	return temps, spins, energies


    
### Recover and process jobs for annealed dynamics 
### Recovers all replicas and seeds 
def process_anneal(timestamp,get_seed=0,get_replicas=None): 
	print("Recovering anneal calculation.")
	### First lets extract all of the different J matrices 
	job_no = io.recover_job_no(timestamp = timestamp)
	print(f"Total jobs: {job_no}")

	seeds = [] ### We will also use a list to store the different seeds
	jobs_by_seed = {} ### Jobs which have the given seed 
	
	replicas_by_job = {}
	
	### Get all the different seeds and replicas for each job with a particular seed and replica 
	for job in range(job_no):
		inputs,data = io.get_results(timestamp=timestamp,run_index=job)
		seed = str(int(inputs['J_seed']))
		replica = int(inputs['replica'])
		
		replicas_by_job[job] = replica

		if seed not in seeds: 
			seeds.append(seed) 
			jobs_by_seed[seed] = [ job ]

		else:
			(jobs_by_seed[seed]).append(job)
    
	spins_stacked = [] 
	energies_stacked = []
    
	temps = [] 
	jobs = jobs_by_seed[seeds[get_seed]]
	for job in jobs:
		if get_replicas is not None and replicas_by_job[job] not in get_replicas:
			continue 
	
		inputs, data = io.get_results(timestamp = timestamp,run_index = job)
		try:
			spins,energies, J = data  
		except:
			print("Legacy dataset, setting energy data to 'None'")
			spins,J = data 
			energies = None   

		temps = inputs['temps']

		seed = str(int(inputs['J_seed']))
		replica= inputs['replica']
		print(f"Job: {job} loaded.") #, replica: {replica}, temps: {temps}, seed: {seed}")
		spins_stacked.append(spins) 
		energies_stacked.append(energies)
    
	### Now we stack the spins by replica 
	print("Stacking data")
	spins_stacked= np.stack(spins_stacked,axis=0) 
	energies_stacked = np.stack(energies_stacked,axis=0)
	temps = np.array(temps)

	return temps, spins_stacked, energies_stacked
	
	
### This method accepts a time stamp, a set of distances, and a chop size and extracts the scale dependent Gaussian noise spectra 
### Assumes spectrum is of annealed type 
### Returns temperatures, frequencies, and spectra 
def process_annealed_spectra(timestamp,distances,chop_size,get_seed=0,get_replicas=None): 
	print("Recovering anneal calculation.")
	### First lets extract all of the different J matrices 
	job_no = io.recover_job_no(timestamp = timestamp)
	print(f"Total jobs: {job_no}")

	seeds = [] ### We will also use a list to store the different seeds
	jobs_by_seed = {} ### Jobs which have the given seed 
	
	replicas_by_job = {}
	Lx,Ly = 0
	### Get all the different seeds and replicas for each job with a particular seed and replica 
	for job in range(job_no):
		inputs,data = io.get_results(timestamp=timestamp,run_index=job)
		seed = str(int(inputs['J_seed']))
		replica = int(inputs['replica'])
		Lx = int(inputs['Lx'])
		Ly = int(inputs['Ly'])
		
		replicas_by_job[job] = replica

		if seed not in seeds: 
			seeds.append(seed) 
			jobs_by_seed[seed] = [ job ]

		else:
			(jobs_by_seed[seed]).append(job)
    
	spins_stacked = [] 
	energies_stacked = []
    
	temps = [] 
	jobs = jobs_by_seed[seeds[get_seed]]
	for job in jobs:
		if get_replicas is not None and replicas_by_job[job] not in get_replicas:
			continue 
	
		inputs, data = io.get_results(timestamp = timestamp,run_index = job)
		try:
			spins,energies, J = data  
		except:
			print("Legacy dataset, setting energy data to None")
			spins,J = data 
			energies = None   

		temps = inputs['temps']

		seed = str(int(inputs['J_seed']))
		replica= inputs['replica']
		print(f"Job: {job}, replica: {replica}, temps: {temps}, seed: {seed}")
		spins_stacked.append(spins) 
		energies_stacked.append(energies)
    
	### Now we stack the spins by replica 
	spins_stacked= np.stack(spins_stacked,axis=0) 
	energies_stacked = np.stack(energies_stacked,axis=0)
	temps = np.array(temps)

	print("Computing local noise.")
	
	noise = nm.calc_local_noise(spins_stacked,distances,Lx,Ly) 
		
	print("Computing spectra.")
	
	ws,spectra = nm.calc_gaussian_spectrum(noise,chop_size)
	
	return temps, ws, spectra
	
	




































