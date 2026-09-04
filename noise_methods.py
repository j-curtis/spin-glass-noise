### Jonathan Curtis
### 12/09/25

import numpy as np
from scipy import stats
import lattice_methods as lm


cumulants_ref = [ 'X','Y','XX','XY','YY','XXX','XXY','XYY','YYY','XXXX','XXXY','XXYY','XYYY','YYYY']


### Computes local magnetic noise a distance d away given a stochastic realization of the spin configuration 
def calc_local_noise(spins,ds,lattice_or_Lx,Ly=None):
	"""Compute the probe field from trajectories with site axis second-to-last.

	A lattice object supplies native geometry.  Passing Lx, Ly remains supported
	for legacy square-lattice callers.
	"""
	### Units and prefactors to keep track
	mu0 = 1. ### vacuum permeability
	muB = 1. ### magnetic moment in bohr magnetons 
	a = 1. ### lattice constant 

	if hasattr(lattice_or_Lx, "magnetic_field_mask_zz"):
		latt = lattice_or_Lx
	else:
		if Ly is None:
			raise ValueError("Ly is required when passing legacy square-lattice dimensions.")
		latt = lm.square_lattice(int(lattice_or_Lx),int(Ly))
	kernels = latt.magnetic_field_mask_zz(ds)

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
	### We assume shape [....,N] where N is the number of time points.
	data = np.asarray(data)
	chop_size = int(chop_size)
	sample_size = 1 if sample_size is None or int(sample_size) <= 0 else int(sample_size)
	if chop_size < 0 or chop_size >= data.shape[-1]:
		raise ValueError("chop_size must leave at least one time point.")
	data_chopped = data[...,chop_size:]
	nblocks = data_chopped.shape[-1]//sample_size
	if nblocks == 0:
		raise ValueError("sample_size is larger than the remaining trajectory.")
	### Block sums preserve the integrated phase while avoiding a dense O(N**2) mask.
	trimmed = data_chopped[...,:nblocks*sample_size]
	return trimmed.reshape(*trimmed.shape[:-1],nblocks,sample_size).sum(axis=-1)
		
		
		
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
	### Put the two echo channels first and average only the sample/replica axis.
	echos = np.moveaxis(echos,-2,0)
	means = np.mean(echos,axis=1)

	moments = np.zeros((14,*means.shape[1:]))
	centered_echos = echos-means[:,None,...]
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
def extract_cumulants_Hahn(noise_sampled,enforce_Z2=False):

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
	### Put the two echo channels first and average only the sample/replica axis.
	echos = np.moveaxis(echos,-2,0)
	means = np.mean(echos,axis=1)
	if enforce_Z2:
		means = np.zeros_like(means)

	moments = np.zeros((14,*means.shape[1:]))
	centered_echos = echos-means[:,None,...]
	moments[:2,...] = means

	for i in range(3):
		moments[2+i,...] = np.mean( (centered_echos[0,...])**(2-i)*(centered_echos[1,...])**i ,axis=0)

	for i in range(4):
		moments[5+i,...] = np.mean( (centered_echos[0,...])**(3-i)*(centered_echos[1,...])**i ,axis=0)
		if enforce_Z2:
			moments[5+i,...] = 0.

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
def thermodynamics(energies,temps,area = 1.,chop_size=None):
	nseeds = len(energies) 
	nreplicas, nTs, nsweeps = energies[0].shape 
	
	### Energy vs. T 
	E_by_lattice = np.zeros((nseeds,nTs)) 
	cV_std_by_lattice = np.zeros((nseeds,nTs))
	cV_eq_by_lattice = np.zeros_like(cV_std_by_lattice)
	
	if chop_size is None: chop_size = int(nsweeps//3)
	
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
	S_by_lattice = np.zeros((nseeds,nTs))
	
	if chop_size<=0: chop_size = int(nsweeps//3)
	
	for i in range(nseeds):
		
		M_by_lattice[i,:] = np.mean(np.abs(mags[i][:,:,-1]),axis=0)
		N_by_lattice[i,:] = np.mean(np.abs(neels[i][:,:,-1]),axis=0)
		stripe_values = np.asarray(stripes[i])
		if stripe_values.ndim != 4 or stripe_values.shape[0] != nreplicas:
			raise ValueError("Each stripe array must have shape (replica, component, temperature, time).")
		### Sum the geometry-native stripe components, then average over replicas.
		S_by_lattice[i,:] = np.mean(np.sum(np.abs(stripe_values[:,:,:,-1]),axis=1),axis=0)

		
	M = np.mean(M_by_lattice,axis=0) 
	N = np.mean(N_by_lattice,axis=0)
	S = np.mean(S_by_lattice,axis=0)
	
	
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
	
	
def annealed_hahn_cumulants(noise,sample_size=1,enforce_Z2=False):
	"""Compute echo cumulants after pooling replicas and disorder realizations.

	``noise`` is a sequence over quenched disorder.  Each element must have a
	leading replica axis and a trailing time axis.  Pooling those leading axes
	before forming moments computes cumulants from the replica/disorder-combined
	annealed MGF; it does not average already-formed per-disorder cumulants.
	"""
	if len(noise) == 0:
		raise ValueError("At least one disorder realization is required.")
	sample_size = 1 if sample_size is None or int(sample_size) <= 0 else int(sample_size)
	sampled = []
	reference_shape = None
	for realization in noise:
		realization = np.asarray(realization,dtype=float)
		if realization.ndim < 2 or realization.shape[0] < 1:
			raise ValueError("Each disorder realization must include a non-empty leading replica axis.")
		reduced = down_sample(realization,realization.shape[-1]//5,sample_size)
		if reference_shape is None:
			reference_shape = reduced.shape[1:]
		elif reduced.shape[1:] != reference_shape:
			raise ValueError("All disorder realizations must have matching non-replica dimensions.")
		sampled.append(reduced)
	pooled_noise = np.concatenate(sampled,axis=0)
	cumulants = extract_cumulants_Hahn(pooled_noise,enforce_Z2=enforce_Z2)
	return echo_times(pooled_noise,sample_size),cumulants


def _fit_power_law(t,y):
	valid = np.isfinite(t) & np.isfinite(y) & (t > 0.) & (y > 0.)
	fitted = np.full_like(y,np.nan,dtype=float)
	if np.count_nonzero(valid) < 2:
		return fitted,np.nan,np.nan,np.nan
	fit = stats.linregress(np.log(t[valid]),np.log(y[valid]))
	fitted[valid] = np.exp(fit.intercept+fit.slope*np.log(t[valid]))
	return fitted,fit.intercept,fit.slope,fit.rvalue


def process_cumulants_av_MGF(noise,sample_size=1,enforce_Z2=False):
	"""Return Hahn cumulants and fits from the annealed replica/disorder MGF."""
	times,cumulants = annealed_hahn_cumulants(noise,sample_size,enforce_Z2=enforce_Z2)
	echo_mean = cumulants[0,...]
	Gamma2 = cumulants[2,...]
	Gamma4 = cumulants[11,...]
	if Gamma2.ndim != 3:
		raise ValueError("Expected cumulants with shape (temperature, distance, delay).")

	fit_shape = Gamma2[...,1:].shape
	Gamma2_fit = {
		'fitted_data':np.full(fit_shape,np.nan),
		'intercepts':np.full(Gamma2.shape[:-1],np.nan),
		'slopes':np.full(Gamma2.shape[:-1],np.nan),
		'rval':np.full(Gamma2.shape[:-1],np.nan),
	}
	Gamma4_fit = {key:value.copy() for key,value in Gamma2_fit.items()}
	for index in np.ndindex(Gamma2.shape[:-1]):
		fit2 = _fit_power_law(times[1:],Gamma2[index][1:])
		fit4 = _fit_power_law(times[1:],-Gamma4[index][1:])
		Gamma2_fit['fitted_data'][index] = fit2[0]
		Gamma4_fit['fitted_data'][index] = -fit4[0]
		for fit_name,value_index in [('intercepts',1),('slopes',2),('rval',3)]:
			Gamma2_fit[fit_name][index] = fit2[value_index]
			Gamma4_fit[fit_name][index] = fit4[value_index]
	return times,Gamma2,Gamma4,Gamma2_fit,Gamma4_fit,echo_mean


def process_cumulants(noise,sample_size=1):
	"""Backwards-compatible five-value wrapper using the correct annealed MGF."""
	return process_cumulants_av_MGF(noise,sample_size)[:5]
	
### Given lattice parameters determines the mean and variance of Jnnn 
def Jnnn_stats(latt):
	pnnn = latt['pnnn']
	Jnnn = latt['Jnnn']

	Jnnn_avg = pnnn*Jnnn
	Jnnn_std = np.sqrt(pnnn*(1.-pnnn))*Jnnn

	return Jnnn_avg,Jnnn_std


### Computes lattice nnn distributions averaged over different seeds
def latt_nnn_dist(lattices):
	nnn_num_dist = [] 

	for i in range(len(lattices)):
		latt = lattices[i]['latt']

		J_matrix = latt.J_matrix
		nnn_matrix = J_matrix.copy()
			
		for site in latt.sites:
			nnn_num = 0
			for nnn in latt.nnns[site]:
				nnn_num += (J_matrix[site,nnn]>0)
                
			nnn_num_dist.append(nnn_num)

	return nnn_num_dist
    
	
	
