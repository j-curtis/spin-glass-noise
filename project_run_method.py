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
	
	sample_step = 1 ### Sample every ___ steps 
	sample_chop = int(nsweeps//5) ### Chop off first __% of each trajectory 

	qea = glauber.calc_frozen_moment(spins,sample_chop,sample_step) 
	
	noise = nm.calc_local_noise(spins,distances,Lx,Ly) 
	
	with open(save_filename, 'wb') as out_file:
        	pickle.dump((J_matrix,energies,magnetization,qea,noise), out_file) ### We store the output spin trajectory, the annealing schedule, and the J config
        	
        	
### Saves a compact output and is low memory usage during operation 
### Forces square lattice 
def run_sims_low_mem(save_filename,L,nsweeps,temps,distances,replica,J_seed):
	L = int(L)
	Lx = L 
	Ly = L 
	J_seed = int(J_seed) 
	replica = int(replica) 

	J_matrix = glauber.nn_coupling_random(-1.,0.5,Lx,Ly,J_seed)

	nns = glauber.nn_indices(Lx,Ly)

	energies, magnetization, qea, noise = glauber.anneal_dynamics_low_mem(J_matrix,nns,nsweeps,temps,distances)
		
	### Due to large memory of spin configurations we will now compute derived observables to save 
	### 1) Energy vs time 
	### 2) Magnetization vs time 
	### 3) Edwards-Anderson OP vs time 
	### 4) Local noise for different distances vs time 
	
	with open(save_filename, 'wb') as out_file:
        	pickle.dump((J_matrix,energies,magnetization,qea,noise), out_file) ### We store the output spin trajectory, the annealing schedule, and the J config
        	
### Saves a compact output and is low memory usage during operation 
### Forces square lattice 
### Implements next-nearest neighbor couplings
def run_sims_low_mem_nnn(save_filename,L,nsweeps,temps,distances,replica,Jnnn,p,J_seed):
	L = int(L)
	nsweeps = int(nsweeps)
	Lx = L 
	Ly = L 
	J_seed = int(J_seed) 
	replica = int(replica) 

	J_matrix, neighbors = glauber.nn_nnn_generate(Lx,Ly,1.,Jnnn,p,J_seed)

	energies, magnetization, neel, qea, noise = glauber.anneal_dynamics_low_mem_nnn(J_matrix,neighbors,nsweeps,temps,distances)
		
	### Due to large memory of spin configurations we will now compute derived observables to save 
	### 1) Energy vs time 
	### 2) Magnetization vs time 
	### 3) Neel order vs time 
	### 4) Edwards-Anderson OP vs time 
	### 5) Local noise for different distances vs time 
	
	with open(save_filename, 'wb') as out_file:
        	pickle.dump((J_matrix,energies,magnetization,neel,qea,noise), out_file) ### We store the output spin trajectory, the annealing schedule, and the J config
        	



### Saves a compact output and is low memory usage during operation 
### Built to work with arbitrary generated lattice objects that are then passed as arguments 
def run_sims_lattice(save_filename,lattice_object,nsweeps,temps,distances,replica):
	L = int(lattice_object.L)
	Lx = L 
	Ly = L 
	
	nsweeps = int(nsweeps)
	replica = int(replica) 

	energies, magnetization, neel, qea, noise = glauber.anneal_dynamics_lattice(lattice_object,nsweeps,temps,distances)
		
	### Due to large memory of spin configurations we will now compute derived observables to save 
	### 1) Energy vs time 
	### 2) Magnetization vs time 
	### 3) Neel order vs time 
	### 4) Edwards-Anderson OP vs time 
	### 5) Local noise for different distances vs time 
	
	with open(save_filename, 'wb') as out_file:
        	pickle.dump((lattice_object,energies,magnetization,neel,qea,noise), out_file) ### We store the output spin trajectory, the annealing schedule, and the lattice
        	
      	
        	
        	
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
		try:
			Lx = int(inputs['Lx'])
			Ly = int(inputs['Ly'])
		except:
			L = int(inputs['L'])
			
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
	
	


### Recover and process jobs for annealed dynamics with compactified data structures 
### Recovers all replicas and seeds 
def process_anneal_observables(timestamp,get_seed=0): 
	print(f"Recovering observables from anneal calculation, timestamp {timestamp}.")
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
		try:
			Lx = int(inputs['Lx'])
			Ly = int(inputs['Ly'])
		except:
			L = int(inputs['L'])
			Lx = L 
			Ly = L 
			
		replicas_by_job[job] = replica

		if seed not in seeds: 
			seeds.append(seed) 
			jobs_by_seed[seed] = [ job ]

		else:
			(jobs_by_seed[seed]).append(job)

	energy = []
	magnetization = [] 
	neel = [] 
	extract_neel = False ### Flag that if activated indicates there is also data on Neel ordering 
	q_ea = []
	noise = [] 
    
	temps = [] 
	distances = [] 
	jobs = jobs_by_seed[seeds[get_seed]]

	for job in jobs:

		inputs, data = io.get_results(timestamp = timestamp,run_index = job)
		if len(data) > 5: extract_neel = True 
		try:
			if not extract_neel:
				J, energy_tmp, magnetization_tmp, q_ea_tmp, noise_tmp = data
			if extract_neel:
				J, energy_tmp, magnetization_tmp, neel_tmp, q_ea_tmp, noise_tmp = data
			energy.append(energy_tmp)
			magnetization.append(magnetization_tmp)
			if extract_neel: neel.append(neel_tmp) 
			q_ea.append(q_ea_tmp)
			noise.append(noise_tmp) 

		except:
			print("Error parsing data.")

		temps = inputs['temps']
		distances = inputs['distances']

		seed = str(int(inputs['J_seed']))
		replica= inputs['replica']
		#print(f"Job: {job} loaded.") 

    
	### Now we stack the spins by replica 
	print("Stacking data")
	energy = np.stack(energy,axis=0)
	magnetization = np.stack(magnetization,axis=0)
	if extract_neel: neel = np.stack(neel,axis=0)
	q_ea = np.stack(q_ea,axis=0)
	noise = np.stack(noise,axis=0) 

	temps = np.array(temps)
	distances = np.array(distances)

	if extract_neel:
		return (Lx,Ly),temps,distances,energy,magnetization,neel,q_ea,noise

	else:
		return (Lx,Ly),temps,distances,energy,magnetization,q_ea,noise




### Recover and process jobs for annealed dynamics with compactified data structures 
### Recovers all replicas and seeds 
### Designed to work for large data sets which could max out the RAM
### Performs down sampling on the noise spectra one by one and then returns this instead 
def process_anneal_memory_safe(timestamp,sample_step=100,get_seed=0): 
	print(f"Recovering observables from anneal calculation, timestamp {timestamp}.")
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
		try:
			Lx = int(inputs['Lx'])
			Ly = int(inputs['Ly'])
		except:
			L = int(inputs['L'])
			Lx = L 
			Ly = L 
			
		replicas_by_job[job] = replica

		if seed not in seeds: 
			seeds.append(seed) 
			jobs_by_seed[seed] = [ job ]

		else:
			(jobs_by_seed[seed]).append(job)

	energy = []
	magnetization = [] 
	neel = [] 
	extract_neel = False ### Flag that if activated indicates there is also data on Neel ordering 
	q_ea = []
	noise = [] 
    
	temps = [] 
	distances = [] 
	jobs = jobs_by_seed[seeds[get_seed]]

	for job in jobs:

		inputs, data = io.get_results(timestamp = timestamp,run_index = job)
		if len(data) > 5: extract_neel = True 
		try:
			if not extract_neel:
				J, energy_tmp, magnetization_tmp, q_ea_tmp, noise_tmp = data
			if extract_neel:
				J, energy_tmp, magnetization_tmp, neel_tmp, q_ea_tmp, noise_tmp = data
			energy.append(energy_tmp)
			magnetization.append(magnetization_tmp)
			if extract_neel: neel.append(neel_tmp) 
			q_ea.append(q_ea_tmp)
			
			### Here we need to down sample the noise to reduce memory demands 
			noise_tmp = nm.down_sample(noise_tmp,chop_size=0,sample_size=sample_step)
			noise.append(noise_tmp) 

		except:
			print("Error parsing data.")

		temps = inputs['temps']
		distances = inputs['distances']

		seed = str(int(inputs['J_seed']))
		replica= inputs['replica']
		#print(f"Job: {job} loaded.") 

    
	### Now we stack the spins by replica 
	print("Stacking data")
	energy = np.stack(energy,axis=0)
	magnetization = np.stack(magnetization,axis=0)
	if extract_neel: neel = np.stack(neel,axis=0)
	q_ea = np.stack(q_ea,axis=0)
	noise = np.stack(noise,axis=0) 

	temps = np.array(temps)
	distances = np.array(distances)

	if extract_neel:
		return (Lx,Ly),temps,distances,energy,magnetization,neel,q_ea,noise

	else:
		return (Lx,Ly),temps,distances,energy,magnetization,q_ea,noise







### This method loads big data sets from qdem 
def load_data_qdem(path,timestamp,get_seed=0,get_replicas=None):
	from demler_tools import file_manager as fm

	### First we point demler_tools to the file location 
	job_no = fm.file_management_local_backend.read_folder_job_no(path+timestamp)

	print(f"Total jobs: {job_no}")
	get_seed=0
	get_replicas = range(10) 
	seeds = [] ### We will also use a list to store the different seeds
	jobs_by_seed = {} ### Jobs which have the given seed 

	replicas_by_job = {}

	### Get all the different seeds and replicas for each job with a particular seed and replica 
	for job in range(job_no):
		inputs,data = fm.file_management_local_backend.read_run_raw_data(path+timestamp,run_index=job)
		if len(data)>5: extract_neel = True 
		seed = str(int(inputs['J_seed']))
		replica = int(inputs['replica'])
		try:
			Lx = int(inputs['Lx'])
			Ly = int(inputs['Ly'])
		except:
			L = int(inputs['L'])
			Lx = L 
			Ly = L 

		replicas_by_job[job] = replica

		if seed not in seeds: 
			seeds.append(seed) 
			jobs_by_seed[seed] = [ job ]

		else:
			(jobs_by_seed[seed]).append(job)


	energy = []
	magnetization = [] 
	neel = [] 
	extract_neel = False ### Flag that if activated indicates there is also data on Neel ordering 
	q_ea = []
	noise = [] 

	temps = [] 
	distances = [] 
	jobs = jobs_by_seed[seeds[get_seed]]

	for job in jobs:

		inputs, data = fm.file_management_local_backend.read_run_raw_data(path+timestamp,run_index=job)
		try:
			if not extract_neel:
				J, energy_tmp, magnetization_tmp, q_ea_tmp, noise_tmp = data
			if extract_neel:
				J, energy_tmp, magnetization_tmp, neel_tmp, q_ea_tmp, noise_tmp = data
			energy.append(energy_tmp)
			magnetization.append(magnetization_tmp)
			if extract_neel: neel.append(neel_tmp) 
			q_ea.append(q_ea_tmp)
			noise.append(noise_tmp) 

		except:
			print("Error parsing data.")

		temps = inputs['temps']
		distances = inputs['distances']

		seed = str(int(inputs['J_seed']))
		replica= inputs['replica']
		#print(f"Job: {job} loaded.") 


	### Now we stack the spins by replica 
	print("Stacking data")
	energy = np.stack(energy,axis=0)
	magnetization = np.stack(magnetization,axis=0)
	if extract_neel: neel = np.stack(neel,axis=0)
	q_ea = np.stack(q_ea,axis=0)
	noise = np.stack(noise,axis=0) 

	temps = np.array(temps)
	distances = np.array(distances)
	print("Data is done being loaded.")

	if extract_neel:
		return (Lx,Ly),temps,distances,energy,magnetization,neel,q_ea,noise

	else:
		return (Lx,Ly),temps,distances,energy,magnetization,None,q_ea,noise
    	
	
### Extraction for new job sets which have multiple different Jnnn sweeps in one job set 
### If get_replicas is not None, we will only extract the set of replicas passed in the list, to save on memory pressure 
def process_nnn_jobs(timestamp,get_replicas=None): 
    print(f"Recovering observables from annealing calculation, timestamp {timestamp}.")
    ### First lets extract all of the different J matrices 
    job_no = io.recover_job_no(timestamp = timestamp)
    print(f"Total jobs: {job_no}")

    jobs_data = {}
    jobs_by_Jnnn = {}  
    jobs_by_pnnn = {} 
    jobs_by_seed = {} 
    jobs_by_replica = {}
    
    replicas_by_job = {}    
    
    params = {}

    ### Get all the different seeds and replicas for each job with a particular seed and replica 
    extract_new_format = False
    for job in range(job_no):
        inputs,data = io.get_results(timestamp=timestamp,run_index=job)
        if len(data) == 6:
            latt, energy, mag, neel, qea, noise = data
            stripes = None
            snapshots = None
        elif len(data) == 8:
            latt, energy, mag, neel, stripes, qea, noise, snapshots = data
            extract_new_format = True
        else:
            raise ValueError(f"Unsupported result tuple length {len(data)} for job {job}.")
        job_data = {'latt':latt, 'energy':energy, 'mag':mag, 'neel':neel, 'stripes':stripes, 'qea':qea, 'noise':noise, 'snapshots':snapshots }

        Jnnn = inputs['Jnnn']
        pnnn = inputs['p']
        seed = int(inputs['J_seed'])
        replica = int(inputs['replica'])
        
        if (get_replicas is not None) and not (replica in get_replicas):
        	continue
        	
        L = int(inputs['L'])
        temps = inputs['temps']
        distances = inputs['distances']
            
        if 'L' not in params.keys():
            params['L'] = L
        if 'temps' not in params.keys():
            params['temps'] = temps 
        if 'distances' not in params.keys():
            params['distances'] = distances 

        replicas_by_job[job] = replica
       	jobs_data[job] = job_data
       	
        if Jnnn not in jobs_by_Jnnn.keys():
            jobs_by_Jnnn[Jnnn] = [ job ]

        else: 
            (jobs_by_Jnnn[Jnnn]).append(job)
        
        if pnnn not in jobs_by_pnnn.keys():
            jobs_by_pnnn[pnnn] = [ job ]

        else: 
            (jobs_by_pnnn[pnnn]).append(job)
   
        if seed not in jobs_by_seed.keys(): 
            jobs_by_seed[seed] = [ job ]

        else:
            (jobs_by_seed[seed]).append(job)         

        if replica not in jobs_by_replica.keys():
            jobs_by_replica[replica] = [ job ] 

        else:
            (jobs_by_replica[replica]).append(job)

    Jnnn_no = len(jobs_by_Jnnn.keys())
    pnnn_no = len(jobs_by_pnnn.keys())
    seed_no = len(jobs_by_seed.keys())
    latt_no = Jnnn_no*pnnn_no*seed_no 
    replica_no = len(jobs_by_replica.keys())

    params['Jnnn_no'] = Jnnn_no
    params['pnnn_no'] = pnnn_no
    params['seed_no'] = seed_no
    params['latt_no'] = latt_no
    params['replica_no'] = replica_no
    
    print(f"Number of Jnnns: {Jnnn_no}")
    print(f"Number of pnnns: {pnnn_no}")
    print(f"Number of seeds: {seed_no}")
    print(f"Number of lattices: {latt_no}")
    print(f"Number of replicas: {replica_no}")

    lattices = [] 
    energies = [] 
    mags = [] 
    neels = [] 
    stripes = []
    qeas = []
    noises = [] 
    snapshots = []
    
    for Jnnn in jobs_by_Jnnn.keys():
        for pnnn in jobs_by_pnnn.keys():
            for seed in jobs_by_seed.keys():
                ### This now identifies a unique lattice which we can save 
                job_list=list( set(jobs_by_Jnnn[Jnnn]) & set(jobs_by_pnnn[pnnn]) & set(jobs_by_seed[seed]))
                latt = jobs_data[job_list[0]]['latt']
                lattices.append({'Jnnn':Jnnn, 'pnnn':pnnn, 'seed':seed, 'latt':latt})
                
                ### We want to turn each of these jobs into an array stacked over replica 
                energies_tmp = []
                mags_tmp = [] 
                neels_tmp = []
                stripes_tmp = []
                qeas_tmp = []
                noises_tmp = [] 
                snapshots_tmp = []
                
                for job in job_list:
                    energies_tmp.append(jobs_data[job]['energy'])
                    mags_tmp.append(jobs_data[job]['mag'])
                    neels_tmp.append(jobs_data[job]['neel'])
                    if extract_new_format: stripes_tmp.append(jobs_data[job]['stripes'])
                    qeas_tmp.append(jobs_data[job]['qea'])
                    noises_tmp.append(jobs_data[job]['noise'])
                    if extract_new_format: snapshots_tmp.append(jobs_data[job]['snapshots'])

                energies.append(np.stack(energies_tmp))
                mags.append(np.stack(mags_tmp))
                neels.append(np.stack(neels_tmp))
                if extract_new_format: stripes.append(np.stack(stripes_tmp))
                qeas.append(np.stack(qeas_tmp))
                noises.append(np.stack(noises_tmp))
                if extract_new_format: snapshots.append(np.stack(snapshots_tmp))
    
    if extract_new_format:
        return params, lattices, energies, mags, neels, stripes, qeas, noises, snapshots

    return params, lattices, energies, mags, neels, qeas, noises

















