import numpy as np
import glauber
import project_run_method as prm
import noise_methods as nm
from scipy import stats

from matplotlib import pyplot as plt 
from matplotlib import cm 
from matplotlib import colors as mclr

def run_rc_defaul():
	plt.rc('font', family = 'Times New Roman')
	plt.rc('font', size = 14)
	plt.rc('text', usetex=True)
	plt.rc('xtick', labelsize=14)
	plt.rc('ytick', labelsize=14)
	plt.rc('axes', labelsize=18)
	plt.rc('lines', linewidth=2.5)
	plt.rc('figure',figsize=(23,10))


def temp_colors(temps):
	### Returns a color scale illustrating a series of temperatures 
	temp_clrs = cm.RdBu((max(temps) - temps[:])/(max(temps) -min(temps))) ### Plot color schemes coding temperature

	return temp_clrs
	
def dist_colors(dists):
	### Returns a color scale illustrating a series of distances 
	dist_clrs = cm.magma_r(0.3 + 0.7* (max(dists) - dists)/(max(dists) - min(dists)) )
	
	return dist_clrs

def plot_schedule(energy, mag, temps,area,replicas = [0],mag_window = 100):
	### Generates plots of energy and magnetization across the annealing schedule 
	### Also generates a plot of energy vs temperature and magnetic susceptibility vs. temperature 
	### Selects just a single replica or all replicas 
	### First we concatenate into a single unified schedule 

	nsweeps = energy.shape[-1]
	ntemps = len(temps)
	temps_cat = np.concatenate([ np.ones(nsweeps)*temps[i] for i in range(ntemps) ] )

	temp_clrs = temp_colors(temps) ### Plot color schemes coding temperature
	text_yval = -1.575
	text_xval = 0.333
	alpha = 0.13

	### Helper functions to extract plot ranges automatically 
	def extract_range(dataset):
		vmax = max(dataset)
		vmin = min(dataset)

		return vmin, vmax
		
		
	### Define the formatter function from https://www.geeksforgeeks.org/data-analysis/formatting-axis-tick-labels-from-numbers-to-thousands-and-millions/
	import matplotlib.ticker as ticker 
	def format_func(value, tick_number):
		return f'{int(value / 1000)}k'

	figs_out = [] 
    
	for r in replicas:
		label = f"annealing_schedule_window={mag_window}_replica={r}"
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

		axs[0].set_ylabel(r'Energy density [$J$]')
		axs[1].set_ylabel(r'Magnetization')
		axs[1].set_xlabel(r'$t$ [MCS]')
		axs[1].xaxis.set_major_formatter(ticker.FuncFormatter(format_func))


		### Extract y ranges 
		mmin,mmax = extract_range(mag_cat)

		### Annotate and color code annealing schedule 
		axs[0].text(-(text_xval+0.7)*nsweeps,text_yval,r'$T/J=$')
		
		
		for i in range(ntemps):
			axs[0].fill_between(np.arange(len(temps_cat))[i*nsweeps:(i+1)*nsweeps],-2,0,facecolor=temp_clrs[i],alpha=alpha)
			axs[0].text(i*nsweeps+nsweeps*text_xval,text_yval,f"{temps[i]:0.2f}")
			axs[1].fill_between(np.arange(len(temps_cat))[i*nsweeps:(i+1)*nsweeps],-1,1,facecolor=temp_clrs[i],alpha=alpha)

		### Set appropriate plot ranges 
		axs[0].set_ylim(-1.5,-.75) ### Energy density should be well captured in this range 

		mrange = max(np.abs(mmin),np.abs(mmax))
		mrange = mrange*1.1
		axs[1].set_ylim(-mrange,mrange)

		for ax in axs: ax.label_outer()

		figs_out.append((fig,label))
		plt.show()

	### Energy vs. T 
	chop_size = int(nsweeps//3)
	label = "internal_energy_heat_capacity_vs_T"
	#E_vs_T = np.mean(energy[:,:,chop_size:],axis=(0,-1))/area
	#c_v = (np.std(energy[:,:,chop_size:],axis=(0,-1))/temps)**2/area
	E_vs_T = np.mean(energy[:,:,-1],axis=0)/area
	c_v = np.std(energy[:,:,chop_size:],axis=(0,-1))**2/(area*temps**2)
	
	fig, ax1 = plt.subplots(1)
	ax1.plot(temps,E_vs_T,'o-',color='purple',label=r'$U$')
	ax1.set_xlabel(r'$T/J$')
	ax1.set_ylabel(r'$U/J$')

	ax2 = ax1.twinx()
	ax2.plot(temps,c_v,'x-',color='black',label=r'$\Delta E^2/T^2$')
	ax2.plot(temps,np.gradient(E_vs_T,temps),'o-',color='gray',label=r'$\frac{dU}{dT}$')
	ax2.set_ylabel(r'$c_V$')

	fig.legend(loc=(0.135,0.8))

	figs_out.append((fig,label))
	plt.show()

	return figs_out 
	
	
def plot_noise_spectra(temps,distances,noise,logx=False,logy=True):

	label_suffix_func = lambda val: 'log' if val else 'lin' 
	label_suffix = "_"+label_suffix_func(logx)+"_"+label_suffix_func(logy)
	
	### Figure sizes 
	fw = 14
	fh = 6
	
	
	ndistances = len(distances)
	ntemps = len(temps)

	### First extract FFT of spectra 
	ws, spectrum = nm.calc_gaussian_spectrum(noise,chop_size=500)

	temp_clrs = temp_colors(temps) ### Plot color schemes coding temperature
	dist_clrs = dist_colors(distances) #cm.Purples(0.5+0.5*(max(distances)-distances[:])/(max(distances)-min(distances)) ) ### Plot color schemes coding distance 

	### Now we generate a series of plots for each distances along with labels 
	figs_noise_vs_freq = [] 
	for i in range(ndistances):
		label = f"noise_vs_freq_d={distances[i]}"+label_suffix

		fig, ax = plt.subplots(1)
		fig.set_figwidth(fw)
		fig.set_figheight(fh)
		
		for j in range(0,ntemps,2):
			ax.plot(ws[1:],spectrum[j,i,1:],color=temp_clrs[j],label=r'$T/J=$'+f"{temps[j]:0.2f}")
		ax.set_xlabel(r'$\omega/2\pi$ [MCS$^{-1}$]')
		ax.set_ylabel(r'$\langle |B(\omega,z)|^2$ [MCS]')
		if logy: ax.set_yscale('log')
		if logx: ax.set_xscale('log')
		ax.legend()
		ax.set_title(r'$d/a=$'+f"{distances[i]:0.0f}")
		ax.set_xlim(1.e-5,5.e-2)
		    
		figs_noise_vs_freq.append((fig,label))
		plt.show()

	### Now we generate a series of plots for each temperature along with labels 
	for i in range(0,ntemps,3):
		label = f"noise_vs_freq_T={temps[i]:0.2f}"+label_suffix

		fig, ax = plt.subplots(1)
		fig.set_figwidth(fw)
		fig.set_figheight(fh)
		
		for j in range(ndistances):
			ax.plot(ws[1:],spectrum[i,j,1:],color=dist_clrs[j],label=r'$d/a=$'+f"{distances[j]:0.0f}")
		ax.set_xlabel(r'$\omega/2\pi$ [MCS$^{-1}$]')
		ax.set_ylabel(r'$\langle |B(\omega,z)|^2$ [MCS]')
		if logy: ax.set_yscale('log')
		if logx: ax.set_xscale('log')
		ax.legend()
		ax.set_title(r'$T/J=$'+f"{temps[i]:0.2f}")
		ax.set_xlim(1.e-5,5.e-2)
		ax.set_ylim(1.e-3,1.e3)
		    
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

	dist_clrs = dist_colors(distances) ### Plot color schemes coding distance 
	for i in range(len(distances)): 
		ax2.plot(temps,static_noise[:,i]/integrated_spectra[:,i],color=dist_clrs[i],label=r'$d/a=$'+f"{distances[i]:0.2f}")
	
	ax2.set_yscale('log')
	ax2.legend()
	ax2.set_ylabel(r'$\mathcal{N}(\omega=0)/\int_\omega \mathcal{N}(\omega)$ [MCS]',color=dist_clrs[0])
	ax2.tick_params(axis='y',labelcolor=dist_clrs[0])
	
	out_fig = fig
	
	label = "frozen_moment"
	plt.show()
	return [(out_fig,label)] 
	



def plot_cumulants(temps,distances,noise,sample_size,z_indxs,temp_indxs,plotting_time_step):
	figs_out = []
	temp_clrs = temp_colors(temps) ### Plot color schemes coding temperature
	
	### Fitting to power laws 
	def fit_cumulant(t,y):
	    
		y_log = np.log(y)
		t_log = np.log(t) 

		fit = stats.linregress(t_log,y_log) 

		return np.exp(fit.intercept +fit.slope*t_log), fit.intercept, fit.slope, fit.rvalue

	### Sample the data down for cumulant calculations
	noise_sampled = nm.down_sample(noise,noise.shape[-1]//5,sample_size)
	cumulants = nm.extract_cumulants_Hahn(noise_sampled)
	echo_times = nm.echo_times(noise_sampled,sample_size)

	### Perform fits 
	fitted_data_Gaussian = np.zeros_like(cumulants[2,:,:,1:])
	fitted_data_Gamma4 = np.zeros_like(cumulants[2,:,:,1:])

	intercepts_Gaussian = np.zeros_like(cumulants[2,:,:,0])
	intercepts_Gamma4 = np.zeros_like(cumulants[2,:,:,0])

	slopes_Gaussian = np.zeros_like(cumulants[2,:,:,0])
	slopes_Gamma4 = np.zeros_like(cumulants[2,:,:,0])

	r_vals_Gaussian = np.zeros_like(cumulants[2,:,:,0])
	r_vals_Gamma4 = np.zeros_like(cumulants[2,:,:,0])

	### We only need to fit the distances/temperatures we are plotting for 
	for i in z_indxs:
		for j in temp_indxs:
			fitted_data_Gaussian[j,i,:],intercepts_Gaussian[j,i], slopes_Gaussian[j,i], r_vals_Gaussian[j,i] = fit_cumulant(echo_times[1:],cumulants[2,j,i,1:])
			fitted_data_Gamma4[j,i,:],intercepts_Gamma4[j,i], slopes_Gamma4[j,i], r_vals_Gamma4[j,i] = fit_cumulant(echo_times[1:],-cumulants[11,j,i,1:])


	### Make plots for each distance and desired cumulant 
	
	### We should make this auto adjust for a number of sample units 
	time_indxs = np.arange(2,len(echo_times),plotting_time_step)

	time_colors = cm.Purples(np.linspace(0.2,1.,len(echo_times)))
	
	for z_indx in z_indxs:
		label=f'gaussian_noise_d={distances[z_indx]:0.2f}'
		fig,ax = plt.subplots()
		for temp_indx in temp_indxs:
			p = slopes_Gaussian[temp_indx,z_indx]
			ax.plot(echo_times[1:],cumulants[2,temp_indx,z_indx,1:],'o-',color=temp_clrs[temp_indx],label=f'$T/J=${temps[temp_indx]:0.2f}')
			ax.plot(echo_times[1:],fitted_data_Gaussian[temp_indx,z_indx,:],'--',color=temp_clrs[temp_indx],label=r'${{\tau}}^{{{p:0.2f}}}$'.format(p=p))
		ax.set_yscale('log')
		ax.set_xscale('log')
		ax.legend()
		ax.set_xlabel(r'Hahn echo time [MCS]')
		ax.set_ylabel(r'Gaussian Hahn echo')
		ax.set_title(f'$d/a =${distances[z_indx]:0.2f}')
		figs_out.append((fig,label))
		plt.show()
    		
		label=f'gamma4_noise_d={distances[z_indx]:0.2f}'
		fig,ax = plt.subplots()
		for temp_indx in temp_indxs:
			p = slopes_Gamma4[temp_indx,z_indx]
			ax.plot(echo_times[1:],-cumulants[11,temp_indx,z_indx,1:],'o-',color=temp_clrs[temp_indx],label=f'$T/J=${temps[temp_indx]:0.2f}')
			ax.plot(echo_times[1:],fitted_data_Gamma4[temp_indx,z_indx,:],'--',color=temp_clrs[temp_indx],label=r'${{\tau}}^{{{p:0.2f}}}$'.format(p=p))
		ax.set_yscale('log')
		ax.set_xscale('log')
		ax.legend()
		ax.set_xlabel(r'Hahn echo time [MCS]')
		ax.set_ylabel(r'$-\Gamma^{(4)}$')
		ax.set_title(f'$d/a =${distances[z_indx]:0.2f}')
		figs_out.append((fig,label))
		plt.show()
		
		label=f'gamma4_vs_T_d={distances[z_indx]:0.2f}'
		fig,ax = plt.subplots()
		for time_indx in time_indxs:
			ax.plot(temps,-cumulants[11,:,z_indx,time_indx],label=r'$ \tau_{{ \rm E }} = $ '+f'{echo_times[time_indx]} [MCS]',color=time_colors[time_indx])
		ax.set_yscale('log')
		ax.set_xlabel(r'$T/J$')
		ax.set_ylabel(r'$-\Gamma^{(4)}$')
		ax.legend()
		figs_out.append((fig,label))
		plt.show()



	return figs_out



def run_annealing_plot_suite(timestamps,sample_size,z_indxs,temp_indxs,plotting_time_step,save_figs=False,window=100):
	### Load data sets 
	print("Loading data sets")
	energy_list = [] 
	mag_list = []
	q_ea_list = []
	noise_list = []

	for timestamp in timestamps:
		(Lx,Ly),temps,distances,energy,mag,q_ea,noise = prm.process_anneal_observables(timestamp)
		ntemps = len(temps)
		print(f"Temperatures per replica: {ntemps}")
		ndists = len(distances)
		nreplicas = energy.shape[0]
		nsweeps = energy.shape[-1]
		print(f"Sweeps per epoch: {nsweeps}")
		energy_list.append(energy)
		mag_list.append(mag)
		q_ea_list.append(q_ea)
		noise_list.append(noise)

	energy = np.concatenate(energy_list)
	mag = np.concatenate(mag_list)
	q_ea = np.concatenate(q_ea_list)
	noise = np.concatenate(noise_list)
	
	### Annealing schedule figure 	
	schedule_fig = pm.plot_schedule(energy,mag,temps,Lx*Ly,mag_window)
	
	### Frozen moment figures 
	frozen_fig = pm.plot_frozen_moment(temps,distances,q_ea,noise)
	
	### Noise spectra figures
	spectra_figs = pm.plot_noise_spectra(temps,distances,noise,logx=True,logy=True) 
	
	### Cumulant figures
	cumulant_figs = pm.plot_cumulants(temps,distances,noise,sample_size,z_indxs,temp_indxs,plotting_time_step)
	
	### Poor mans approach to figure saving for now
	fig_directory_path = "/home/jcurtis/Projects/SpinGlassNoise/figs/" + "".join([ timestamp[-5:]+"_" for timestamp in timestamps ])[:-1]+"/"

	if not os.path.isdir(fig_directory_path):
		os.makedirs(fig_directory_path)

		
	if save_figs:
		for fig,label in schedule_fig:
			fig_path = fig_directory_path + label+".pdf"
			fig.savefig(fig_path,bbox_inches='tight') 

		for fig,label in frozen_fig:
			fig_path = fig_directory_path + label+".pdf"
			fig.savefig(fig_path,bbox_inches='tight')

		for fig,label in spectra_figs:
			fig_path = fig_directory_path + label+".pdf"
			fig.savefig(fig_path,bbox_inches='tight')

		for fig,label in cumulant_figs:
			fig_path = fig_directory_path + label+".pdf"
			fig.savefig(fig_path,bbox_inches='tight')


	
	
