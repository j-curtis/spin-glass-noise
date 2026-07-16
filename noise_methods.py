### Jonathan Curtis
### 12/09/25

import numpy as np
import glauber


cumulants_ref = [ 'X','Y','XX','XY','YY','XXX','XXY','XYY','YYY','XXXX','XXXY','XXYY','XYYY','YYYY']


### Computes local magnetic noise a distance d away given a stochastic realization of the spin configuration 
def calc_local_noise(spins,ds,Lx,Ly):
	### Units and prefactors to keep track
	mu0 = 1. ### vacuum permeability
	muB = 1. ### magnetic moment in bohr magnetons 
	a = 1. ### lattice constant 

	### Computes local magnetic noise
	### Assumes spin qubit is a distance d away in the center of the sample 

	### The magnetization in the z direction a distance d away is given in terms of the spins as 
	### B_z(t) = mu_0 mu_B mu_mat/(4 pi) sum_j S_j(t) (2d^2 - R_j^2)/(R_j^2 + d^2)^(3/2) 

	X,Y = np.meshgrid(np.arange(Lx)-Lx//2, np.arange(Ly)-Ly//2,indexing='ij')

	R = np.sqrt(X**2 + Y**2) 

	R = R.ravel() 

	kernels = (2.*ds[None,:]**2 - R[:,None]**2)/( R[:,None]**2 + ds[None,:]**2)**2.5 ### Has shape [Nspins, Nds] 

	noise = np.tensordot(spins,kernels,axes=[-2,0]) * mu0*muB/(4.*np.pi*a**3)
	
	### Now we swap the last two axes so that the time axis is last as always 
	noise = np.swapaxes(noise,-1,-2) 

	return noise 


### Computes the echo phase for a given noise trajectory 
### times = (t0,...,tf) where t0 and tf are the initial and final pi/2 pulses and intermediate values are optional pi echo times (as integers)
def calc_echo_phase(noise,times):
	t0 = times[0]
	tpis = times[1:-1]
	tf = times[-1] 

	filter_func = np.zeros(noise.shape[-1])
	filter_func[t0:tf] = 1. 

	for i in tpis:
		filter_func[t0:i] *= -1 

	return np.tensordot(noise,filter_func,axes=[-1,0]) 

    
### Computes the linear spectrum averaged over replica 
### Optionally subtracts time-average first 
def calc_gaussian_spectrum(noise,chop_size,center=False):
	### Returns the FFT of the noise and the frequencies after chopping some initial transient response 
	
	data = noise[...,chop_size:] 
	
	if center: data = data - np.mean(data,axis=-1,keepdims=True)
	
	fft_data = np.fft.rfft(data,axis=-1)
	
	ws = np.fft.rfftfreq(data.shape[-1])
	
	spectrum = np.abs(fft_data)**2
	
	### Average the spectrum over replicas 
	return ws, np.mean(spectrum,axis=0)
	
	
	
### This function will compute the cumulant function for a given pulse sequence averaged over replicas 
def calc_cumulant(noise,times):
	### Chop size is implicitly specified in the times arguments 
	
	cumulants = np.exp(1.j*calc_echo_phase(noise,times) )
	
	### Now average over replicas 
	return np.mean(cumulants,axis=0) 
	
 
 
### This function will down sample the noise and perform averaging over blocks which allows the reduction of data size as well as enables direct calculation of echo phases 
def down_sample(data,chop_size,sample_size):
	### First we chop the data 
	### We assume shape [....,N] where N is the number of time points 
	data_chopped = data[...,chop_size:] 
	
	### Next we generate matrices for masking the data to implement the sample averaging 
	ntimes = data_chopped.shape[-1] 
	mask_matrix = np.zeros((ntimes,ntimes//sample_size))
	for j in range(mask_matrix.shape[-1]):
		mask_matrix[j*sample_size:(j*sample_size+sample_size),j] = 1. 
		
	### Time points in original MCS units 
	times = np.arange(mask_matrix.shape[-1])*sample_size
	
	return np.tensordot(data_chopped,mask_matrix,axes=[-1,0])
		
		
		
### Given the down-sampled noise this computes the first four cumulants of the noise as a function of echo delay 
### Built on Ramsey sequence and the needed higher echos  
def extract_cumulants_Ramsey(noise_sampled):

	### Methods for processesing the cumulants of the Ramsey echo 

	nsampled = noise_sampled.shape[-1]
	ndelays = nsampled//2

	delays = np.arange(ndelays)
	filters = np.zeros((2,ndelays,nsampled))
	echos = np.zeros((*noise_sampled.shape[:-1],2,ndelays))

	for i in range(ndelays):
		filters[0,i,:i] = 1.
		filters[1,i,i:2*i] = 1.

	echos = np.tensordot( noise_sampled,filters,axes=[-1,-1])
	echos = np.rollaxis(echos, -2, 1)
	means = np.mean(echos,axis=0)

	moments = np.zeros((14,*means.shape[1:]))
	centered_echos = echos - means[None,...]
	moments[:2,...] = means

	for i in range(3):
		moments[2+i,...] = np.mean( (centered_echos[0,...])**(2-i)*(centered_echos[1,...])**i ,axis=0)

	for i in range(4):
		moments[5+i,...] = np.mean( (centered_echos[0,...])**(3-i)*(centered_echos[1,...])**i ,axis=0)

	for i in range(5):
		moments[9+i,...] = np.mean( (centered_echos[0,...])**(4-i)*(centered_echos[1,...])**i ,axis=0)

	cumulants = moments.copy() 

	### Only at fourth order are the cumulants different from the central moments 
	cumulants[9,...] = moments[9,...] - 3.*(moments[2,...])**2 
	cumulants[10,...] = moments[10,...] - 3.*moments[2,...]*moments[3,...]
	cumulants[11,...] = moments[11,...] - 2.*moments[3,...]**2 - moments[2,...]*moments[4,...] 
	cumulants[12,...] = moments[12,...] - 3.*moments[4,...]*moments[3,...]
	cumulants[13,...] = moments[13,...] - 3.*(moments[4,...])**2

	return cumulants 
    
### Given the down-sampled noise this computes the first four cumulants of the noise as a function of echo delay 
### Built on Hahn sequence and the needed higher echos  
def extract_cumulants_Hahn(noise_sampled):

	### Methods for processesing the cumulants of the Ramsey echo 

	nsampled = noise_sampled.shape[-1]
	ndelays = nsampled//4 

	delays = np.arange(ndelays)
	filters = np.zeros((2,ndelays,nsampled))
	echos = np.zeros((*noise_sampled.shape[:-1],2,ndelays))

	for i in range(1,ndelays):
		filters[0,i,:2*i] = np.sign(np.arange(2*i) -i+0.5 )
		filters[1,i,2*i:4*i] = np.sign(np.arange(2*i) -i+0.5 )

	echos = np.tensordot( noise_sampled,filters,axes=[-1,-1])
	echos = np.rollaxis(echos, -2, 1)
	means = np.mean(echos,axis=0)

	moments = np.zeros((14,*means.shape[1:]))
	centered_echos = echos - means[None,...]
	moments[:2,...] = means

	for i in range(3):
		moments[2+i,...] = np.mean( (centered_echos[0,...])**(2-i)*(centered_echos[1,...])**i ,axis=0)

	for i in range(4):
		moments[5+i,...] = np.mean( (centered_echos[0,...])**(3-i)*(centered_echos[1,...])**i ,axis=0)

	for i in range(5):
		moments[9+i,...] = np.mean( (centered_echos[0,...])**(4-i)*(centered_echos[1,...])**i ,axis=0)

	cumulants = moments.copy() 

	### Only at fourth order are the cumulants different from the central moments 
	cumulants[9,...] = moments[9,...] - 3.*(moments[2,...])**2 
	cumulants[10,...] = moments[10,...] - 3.*moments[2,...]*moments[3,...]
	cumulants[11,...] = moments[11,...] - 2.*moments[3,...]**2 - moments[2,...]*moments[4,...] 
	cumulants[12,...] = moments[12,...] - 3.*moments[4,...]*moments[3,...]
	cumulants[13,...] = moments[13,...] - 3.*(moments[4,...])**2

	return cumulants 		
		
def echo_times(noise_sampled,sample_times):
	nsampled = noise_sampled.shape[-1]
	ndelays = nsampled//4 

	delays = np.arange(ndelays)
	
	return delays*sample_times 
	
	
	
	
### This method computes the thermodynamics of a data set 
### The data is assumed to correspond to one set of lattice parameters but potentially multiple seeds (disorder realizations)
### We therefore process first the data for each disorder and then average last over seed
### The data is assumed to be a shape of the form 
### list[ energies[replica, temp, time] ] where the list runs over each lattice seed 
def thermodynamics(energies,temps,area = 1.,chop_size=0):
	nseeds = len(energies) 
	nreplicas, nTs, nsweeps = energies[0].shape 
	
	### Energy vs. T 
	E_by_lattice = np.zeros((nseeds,nTs)) 
	cV_std_by_lattice = np.zeros((nseeds,nTs))
	cV_eq_by_lattice = np.zeros_like(cV_std_by_lattice)
	
	if chop_size<=0: chop_size = int(nsweeps//3)
	
	for i in range(nseeds):
		
		E_by_lattice[i,:] = np.mean(energies[i][:,:,-1],axis=0)/area
		cV_std_by_lattice[i,:] = np.std(energies[i][:,:,chop_size:],axis=(0,-1))**2/(area*temps**2)

	for i in range(nseeds):
		cV_eq_by_lattice[i,:] = np.gradient(E_by_lattice[i,:],temps) 
		
	E_vs_T = np.mean(E_by_lattice,axis=0) 
	cV_std = np.mean(cV_std_by_lattice,axis=0)
	cV_eq = np.mean(cV_eq_by_lattice,axis=0) 
	
	
	return E_vs_T,cV_std,cV_eq
	
	
### This method computes the various order parameters of the data set 
### The data is assumed to correspond to one set of lattice parameters but potentially multiple seeds (disorder realizations)
### We therefore process first the data for each disorder and then average last over seed
### The data is assumed to be a shape of the form 
### list[ OP[dof, replica, temp, time] ] where the list runs over each lattice seed and dof is either trivial (for magnetization and Neel) or is the component of stripe order (0,1) for the stripe orders
### All OPs are Z2 and this returns the average of the modulus of the OP over replica and disorder 
def order_parameters(mags,neels,stripes,chop_size=0):
	 
	nseeds = len(mags) 
	nreplicas, nTs, nsweeps = mags[0].shape 
	
	### Magnetization
	M_by_lattice = np.zeros((nseeds,nTs))
	N_by_lattice = np.zeros((nseeds,nTs))
	Sx_by_lattice = np.zeros((nseeds,nTs))
	Sy_by_lattice = np.zeros((nseeds,nTs)) 
	
	if chop_size<=0: chop_size = int(nsweeps//3)
	
	for i in range(nseeds):
		
		M_by_lattice[i,:] = np.mean(np.abs(mags[i][:,:,-1]),axis=0)
		N_by_lattice[i,:] = np.mean(np.abs(neels[i][:,:,-1]),axis=0)
		Sx_by_lattice[i,:] = np.mean(np.abs(stripes[i][0,:,:,-1]),axis=0)
		Sy_by_lattice[i,:] = np.mean(np.abs(stripes[i][1,:,:,-1]),axis=0)

		
	M = np.mean(M_by_lattice,axis=0) 
	N = np.mean(N_by_lattice,axis=0)
	S = np.mean(Sx_by_lattice + Sy_by_lattice,axis=0) 
	
	
	return M,N,S
	
	
### This method computes the Gaussian noise spectrum for the data set 
### The data is assumed to correspond to one set of lattice parameters but potentially multiple seeds (disorder realizations)
### We therefore process first the data for each disorder and then average last over seed
### The data is assumed to be a shape of the form 
### list[ noise[..., time] ] where the list runs over each lattice seed
### By default it uses the centered spectrum which subtracts the time-average at omega = 0 
def calc_noise_spectrum(noise,chop_size=0,center=True):
	nseeds = len(noise)
	
	noise_shape = noise[0].shape 
	nsweeps = noise_shape[-1]
	
	noise_by_lattice = []
	
	
	if chop_size<=0: chop_size = int(1e3)
	
	for i in range(nseeds):
		ws,noise_out = calc_gaussian_spectrum(noise[i],chop_size,center=center)
		noise_by_lattice.append(noise_out)
		
	noise_by_lattice = np.stack(noise_by_lattice,axis=0) 
	spectrum = np.mean(noise_by_lattice,axis=0) 
	
	return ws,spectrum
	
	
### Processes the cumulants and then averages over the lattice disorder realizations 
### Assumed data of the form 
### list[noises[....,time] ] where list runs over each lattice seed 
### Returns the desired (2,2) fourth cumulant and a fitted time dependence after disorder averaging and the down-sampled echo times 
### Noise is down-sampled by the specified amount which defaults to zero (very costly) 
def process_cumulants(noise,sample_size=0):
	nseeds = len(noise)
	
	Gamma2_by_lattice = [] 
	Gamma4_by_lattice = [] 
	echo_times = None 
	
	for i in range(nseeds):
	
		### Sample the data down for cumulant calculations
		noise_sampled = down_sample(noise[i],noise[i].shape[-1]//5,sample_size)
		cumulants = extract_cumulants_Hahn(noise_sampled)
		echo_times = echo_times(noise_sampled,sample_size)

		Gamma2_by_lattice.append(cumulants[2,...])
		Gamma4_by_lattice.append(cumulants[11,...])
		
	### Now we restack and average over disorder 
	Gamma2 = np.mean(np.stack(Gamam2_by_lattice,axis=0),axis=0)
	Gamma4 = np.mean(np.stack(Gamma4_by_lattice,axis=0),axis=0)
	
	### Now we fit the data
		
	### Fitting to a single power law
	def fit_cumulant(t,y):
	    
		y_log = np.log(y)
		t_log = np.log(t) 

		fit = stats.linregress(t_log,y_log) 

		return np.exp(fit.intercept +fit.slope*t_log), fit.intercept, fit.slope, fit.rvalue


	### Perform fits 
	fitted_data_Gamma2 = np.zeros_like(Gamma2)
	fitted_data_Gamma4 = np.zeros_like(Gamma4)

	intercepts_Gamma2 = np.zeros_like(Gamma2[...,0])
	intercepts_Gamma4 = np.zeros_like(Gamma4[...,0])

	slopes_Gamma2 = np.zeros_like(Gamma2[...,0])
	slopes_Gamma4 = np.zeros_like(Gamma4[...,0])

	r_vals_Gamma4 = np.zeros_like(Gamma2[...,0])
	r_vals_Gamma4 = np.zeros_like(Gamma4[...,0])

	### Infer shape of z and temperature indices 
	ndists, ntemps, _ = Gamma2.shape 

	for i in range(ndists):
		for j in range(ntemps):
			fitted_data_Gamma2[j,i,:],intercepts_Gamma2[j,i], slopes_Gamma2[j,i], r_vals_Gamma2[j,i] = fit_cumulant(echo_times[1:],Gamma2[...,1:])
			fitted_data_Gamma4[j,i,:],intercepts_Gamma4[j,i], slopes_Gamma4[j,i], r_vals_Gamma4[j,i] = fit_cumulant(echo_times[1:],-Gamma4[...,1:) ### We expect Gamma4 <0 so we fit the negative value to a power law 
			fitted_data_Gamma4[j,i,:] *= -1 ### Flip the sign back 
			intercepts_Gamma4[j,i] *= -1. ### Intercept also flips back 

	### We return the cumulants, the times, and the fit results 
	Gamma2_fit = {'fitted_data':fitted_data_Gamma2, 'intercepts':intercepts_Gamma2, 'slopes':slopes_Gamma2, 'rval':r_vals_Gamma2}
	Gamma4_fit = {'fitted_data':fitted_data_Gamma4, 'intercepts':intercepts_Gamma4, 'slopes':slopes_Gamma4, 'rval':r_vals_Gamma4}


	return echo_times, Gamma2, Gamma4, Gamma2_fit, Gamma4_fit 
	
	

