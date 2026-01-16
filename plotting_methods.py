import numpy as np
import glauber
import project_run_method as prm
import noise_methods as nm

from matplotlib import pyplot as plt 
from matplotlib import cm 
from matplotlib import colors as mclr

plt.rc('font', family = 'Times New Roman')
plt.rc('font', size = 14)
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=18)
plt.rc('lines', linewidth=2.5)
plt.rc('figure',figsize=(23,10))

def plot_schedule(energy, mag, temps,area,replicas = [0],mag_window = 100):
	### Generates plots of energy and magnetization across the annealing schedule 
	### Selects just a single replica or all replicas 
	### First we concatenate into a single unified schedule 

	nsweeps = energy.shape[-1]
	ntemps = len(temps)
	temps_cat = np.concatenate([ np.ones(nsweeps)*temps[i] for i in range(ntemps) ] )

	temp_clrs = cm.RdBu((max(temps) - temps[:])/(max(temps) -min(temps))) ### Plot color schemes coding temperature
	text_yval = -1.575
	text_xval = 0.333
	alpha = 0.13

	### Helper functions to extract plot ranges automatically 
	def extract_range(dataset):
		vmax = max(dataset)
		vmin = min(dataset)

		return vmin, vmax

	figs_out = [] 
    
	for r in replicas:
		energy_cat = np.concatenate(energy[r,...]/area,axis=0)
		mag_cat = np.concatenate(mag[r,...],axis=0)
		if mag_window >0: mag_cat = glauber.moving_avg(mag_cat,mag_window)

		### First we generate the plots and subfigures 
		fig = plt.figure()
		grid = fig.add_gridspec(2,hspace=.15)
		axs = grid.subplots(sharex=True)

		### Plot the data sets and label axes 
		axs[0].plot(energy_cat,color='purple')
		axs[1].plot(mag_cat,color='orange')
		axs[1].axhline(0.,linestyle='dashed',color='gray')

		axs[0].set_ylabel(r'Energy [$J$]')
		axs[1].set_ylabel(r'Magnetization')
		axs[1].set_xlabel(r'Time steps')


		### Extract y ranges 
		mmin,mmax = extract_range(mag_cat)

		### Annotate and color code annealing schedule 
		axs[0].text(-(1-text_xval)*nsweeps,text_yval,r'$T/J=$',fontsize='x-large')
		for i in range(ntemps):
			axs[0].fill_between(np.arange(len(temps_cat))[i*nsweeps:(i+1)*nsweeps],-2,0,facecolor=temp_clrs[i],alpha=alpha)
			axs[0].text(i*nsweeps+nsweeps*text_xval,text_yval,f"{temps[i]:0.2f}",fontsize='x-large')
			axs[1].fill_between(np.arange(len(temps_cat))[i*nsweeps:(i+1)*nsweeps],-1,1,facecolor=temp_clrs[i],alpha=alpha)

		### Set appropriate plot ranges 
		axs[0].set_ylim(-1.5,-.75) ### Energy density should be well captured in this range 

		mrange = max(np.abs(mmin),np.abs(mmax))
		mrange = mrange*1.1
		axs[1].set_ylim(-mrange,mrange)

		for ax in axs: ax.label_outer()

		figs_out.append(fig)
		plt.show()

	return figs_out 
	
	
def plot_noise_spectra(temps,distances,noise):

	ndistances = len(distances)
	ntemps = len(temps)

	### First extract FFT of spectra 
	ws, spectrum = nm.calc_gaussian_spectrum(noise,chop_size=500)

	temp_clrs = cm.RdBu((max(temps) - temps[:])/(max(temps) -min(temps))) ### Plot color schemes coding temperature
	dist_clrs = cm.Purples(0.5+0.5*(max(distances)-distances[:])/(max(distances)-min(distances)) ) ### Plot color schemes coding distance 

	### Now we generate a series of plots for each distances along with labels 
	figs_noise_vs_freq = [] 
	for i in range(ndistances):
		label = f"noise_vs_freq_d={distances[i]}"

		fig, ax = plt.subplots(1)
		for j in range(0,ntemps,2):
			ax.plot(ws,spectrum[j,i,:],color=temp_clrs[j],label=r'$T/J=$'+f"{temps[j]:0.2f}")
		ax.set_xlabel(r'$\omega/2\pi$ [$\Delta t^{-1}$]')
		ax.set_ylabel(r'$\langle |B(\omega,z)|^2$ [$\Delta t$]')
		ax.set_yscale('log')
		#plt.xscale('log')
		ax.legend()
		ax.set_title(r'$d/a=$'+f"{distances[i]:0.0f}")
		ax.set_xlim(0.,5.e-2)
		    
		figs_noise_vs_freq.append((fig,label))
		plt.show()

	### Now we generate a series of plots for each temperature along with labels 
	for i in range(0,ntemps,3):
		label = f"noise_vs_freq_T={temps[i]:0.2f}"

		fig, ax = plt.subplots(1)
		for j in range(ndistances):
			ax.plot(ws,spectrum[i,j,:],color=dist_clrs[j],label=r'$d/a=$'+f"{distances[j]:0.0f}")
		ax.set_xlabel(r'$\omega/2\pi$ [$\Delta t^{-1}$]')
		ax.set_ylabel(r'$\langle |B(\omega,z)|^2$ [$\Delta t$]')
		ax.set_yscale('log')
		#plt.xscale('log')
		ax.legend()
		ax.set_title(r'$T/J=$'+f"{temps[i]:0.2f}")
		ax.set_xlim(0.,5.e-2)
		ax.set_ylim(1.e-4,1.e2)
		    
		figs_noise_vs_freq.append((fig,label))
		plt.show()


	return figs_noise_vs_freq	
	
	
def plot_frozen_moment(temps,distances,q_ea,noise):
	### First we extract the noise spectra 
	ws, spectra = nm.calc_gaussian_spectrum(noise,chop_size=500)
	integrated_spectra = np.trapz(spectra,ws,axis=-1)
	static_noise = spectra[:,:,0]

	### Now we plot the frozen moment squared and the noise at zero frequency normalized by the total power spectral denstiy for different distances as a function of temperature 
	fig,ax1 = plt.subplots() 

	color = 'blue'
	ax1.errorbar(temps,np.mean(q_ea,axis=0),np.std(q_ea,axis=0),None,'o-',color=color)
	ax1.set_ylim(0.,1.)
	ax1.set_xlabel(r'$T/J$')
	ax1.set_ylabel(r'$q_{EA}$',color=color)
	ax1.tick_params(axis='y',labelcolor=color)

	ax2 = ax1.twinx()

	dist_clrs = cm.Purples(0.25+0.75*(max(distances)-distances[:])/(max(distances)-min(distances)) ) ### Plot color schemes coding distance 
	for i in range(len(distances)):
	    
		ax2.plot(temps,static_noise[:,i]/integrated_spectra[:,i],color=dist_clrs[i],label=r'$d/a=$'+f"{distances[i]:0.2f}")
	ax2.set_yscale('log')
	ax2.legend()
	ax2.set_ylabel(r'$\mathcal{N}(\omega=0)/\int_\omega \mathcal{N}(\omega)$',color=dist_clrs[0])
	ax2.tick_params(axis='y',labelcolor=dist_clrs[0])
	
	out_fig = fig
	plt.show()
	return out_fig 
	






















	
	
