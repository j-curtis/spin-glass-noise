### Jonathan Curtis
### 12/09/25

import numpy as np
import glauber


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
def calc_gaussian_spectrum(noise,chop_size):
	### Returns the FFT of the noise and the frequencies after chopping some initial transient response 
	
	data = noise[...,chop_size:] 
	
	fft_data = np.fft.rfft(data,axis=-1)
	
	ws = np.fft.rfftfreq(data.shape[-1])
	
	spectrum = np.abs(fft_data)**2
	
	### Average the spectrum over replicas 
	return ws, np.mean(spectrum,axis=0)
	
