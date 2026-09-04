### Lattice geometry, topology, interaction constants, and geometry-dependent observables
### Jonathan Curtis

from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import csr_matrix, issparse, isspmatrix_lil, lil_matrix


__version__ = "0.1.0"


class Lattice(ABC):
	"""Base class for a two-dimensional Bravais lattice with a finite basis."""

	geometry = None
	nbasis = None ### !!! Codex: Does it make more sense assign this to a default of 1? 
	primitive_vectors = None
	basis_vectors = None

	def __init__(self,Lx,Ly=None,periodic=True,origin_site=0):
		self.symmetry_tol = 1.e-7
		self.Lx = int(Lx)
		self.Ly = self.Lx if Ly is None else int(Ly)
		if self.Lx <= 0 or self.Ly <= 0:
			raise ValueError("Lx and Ly must be positive.")
		self.L = self.Lx  # Backwards compatibility with square-lattice jobs. ### !!! Codex: do we really need this?
		self.periodic = bool(periodic)
		self.N = self.Lx*self.Ly*self.nbasis ### !!! Codex: do I understand correctly that if one runs the base method, self.nbasis is initialized to None and this will be ill-defined? Also consider adding also an alias attribute for N with name nspins or nsites which also might be often referenced 
		self.sites = np.arange(self.N)

		### !!! Codex: please add a very brief 1 line comment for each of these explaining what they are, in my style of writing
		self.cell_indices = np.zeros((self.N,2),dtype=int)
		self.sublattice = np.zeros(self.N,dtype=int)
		self.positions = np.zeros((self.N,2),dtype=float)
		self._build_coordinates()

		origin_site = int(origin_site)
		if origin_site < 0 or origin_site >= self.N:
			raise IndexError("origin_site is outside the lattice.")
		self.origin_site = origin_site
		self.coordinate_offset = self.positions[origin_site].copy()
		### Translate the declared origin site to exactly (0, 0); all relative
		### positions remain fixed by the subclass Bravais and basis vectors.
		self.positions -= self.coordinate_offset[None,:]
		self.origin_xy = np.zeros(2,dtype=float)
		self.X = self.positions[:,0]
		self.Y = self.positions[:,1]
		self.R = np.sqrt(self.X**2+self.Y**2)

		self.neighbor_shells = self._build_neighbor_shells()
		self._validate_shell_storage()
		### Compatibility names used by existing dynamics and saved data.
		self.nns = self.neighbor_shells["nn"]
		self.nnns = self.neighbor_shells.get("nnn",self._empty_shell())
		self.partners = [
			sorted(set().union(*(shell[i] for shell in self.neighbor_shells.values())))
			for i in self.sites ### !!! Codex: this is a bit unclear, please add a comment here explaining what this does, succinctly in 1 line.
		]
		### Couplings are assembled in LIL form and compressed to CSR before dynamics.
		### Both representations scale with the number of bonds rather than N**2.
		self.J_matrix = lil_matrix((self.N,self.N),dtype=np.float64)
		self.coupling_specs = []
		self.order_names = self.order_parameter_masks()[0] ### !!! Codex: consider renaming to "order_parameters" to be more clear about the name

		self.seed = None
		self.rng = np.random.default_rng()
		self._interaction_colors = None
		self._interaction_color_classes = None
		self.validate_topology()

	def _build_coordinates(self):
		primitive = np.asarray(self.primitive_vectors,dtype=float)
		basis = np.asarray(self.basis_vectors,dtype=float)
		if primitive.shape != (2,2):
			raise ValueError("primitive_vectors must have shape (2, 2).")
		if basis.shape != (self.nbasis,2):
			raise ValueError(f"basis_vectors must have shape ({self.nbasis}, 2).")
		for y in range(self.Ly):
			for x in range(self.Lx):
				for sub in range(self.nbasis):
					i = self.coordinate_to_index((x,y,sub),wrap=False)
					self.cell_indices[i] = (x,y)
					self.sublattice[i] = sub
					self.positions[i] = x*primitive[0]+y*primitive[1]+basis[sub]

	@abstractmethod
	def _build_neighbor_shells(self):
		"""Return ``{shell_name: per-site neighbor lists}`` for this geometry."""

	@abstractmethod
	def order_parameter_masks(self):
		"""Return a tuple of names and an array with shape (norders, N)."""

	def _empty_shell(self):
		return [[] for _ in self.sites]

	### !!! Codex: once again, I am not sure about this name. Wouldn't "_coordinate_to_index" be an equivalent name? It is not necessary that it only be used on a candidate shifted coordinate? If no, then the name should be more general
	def _candidate_coordinate_to_index(self,x,y,sub):
		"""Convert an already-shifted candidate cell coordinate to a site index."""
		if self.periodic:
			return self.coordinate_to_index((x,y,sub),wrap=True)
		if x < 0 or x >= self.Lx or y < 0 or y >= self.Ly:
			return None
		return self.coordinate_to_index((x,y,sub),wrap=False)

	### !!! Codex: what is the purpose of this method? please add a brief description 
	def _indices_from_specs(self,site,specs):
		neighbors = []
		for spec in specs:
			j = self._candidate_coordinate_to_index(*spec)
			if j is not None and j != site and j not in neighbors:
				neighbors.append(j)
		return neighbors

	def index_to_coordinate(self,i):
		i = int(i)
		if i < 0 or i >= self.N:
			raise IndexError(f"site index {i} is outside the lattice.")
		x,y = self.cell_indices[i]
		if self.nbasis == 1:
			return int(x),int(y) ### !!! Codex: does it make more sense to always return of the same signature: int(x), int(y), int(sublattice)? If nbasis = 1 then the last index can be always put to one, but the shape retained is consistent. Please consider changing this throughout 
		return int(x),int(y),int(self.sublattice[i])

	def coordinate_to_index(self,r,wrap=True):
		if len(r) == 2:
			x,y = r
			sub = 0
		elif len(r) == 3:
			x,y,sub = r
		else:
			raise ValueError("A lattice coordinate must contain (x, y) or (x, y, sublattice).")
		x,y,sub = int(x),int(y),int(sub)
		if sub < 0 or sub >= self.nbasis:
			raise ValueError(f"sublattice must lie between 0 and {self.nbasis-1}.")
		if wrap:
			x %= self.Lx
			y %= self.Ly
		elif x < 0 or x >= self.Lx or y < 0 or y >= self.Ly:
			raise IndexError("cell coordinate is outside the lattice.")
		return sub+self.nbasis*(x+self.Lx*y)

	@property
	def site_centroid(self):
		"""Arithmetic centroid of the finite set of sites; not the coordinate origin."""
		return np.mean(self.positions,axis=0)

	def set_seed(self,seed):
		self.seed = seed
		self.rng = np.random.default_rng(self.seed)

	def _invalidate_coloring(self):
		self._interaction_colors = None
		self._interaction_color_classes = None

	def _require_shell(self,shell):
		if shell not in self.neighbor_shells:
			allowed = ", ".join(sorted(self.neighbor_shells))
			raise ValueError(f"Given shell '{shell}' lies outside the allowed shells: {allowed}.")
		return self.neighbor_shells[shell]

	### !!! Codex: as I understand it, this method assumes only a Bernoulli type distribution of couplings, in the case of random couplings. This is not guaranteed, and sometimes, e.g. Gaussian distributions will be desired. The method should ideally be more general or explicitly flag the limitation in terms of distributions supported
	def set_shell_J(self,shell,J,values=None,probabilities=None):
		"""Set one symmetric coupling per unique bond in an arbitrary shell."""
		self._ensure_writeable_couplings()
		if values is None:
			values = np.atleast_1d(np.asarray(J,dtype=float))
		else:
			values = np.atleast_1d(np.asarray(values,dtype=float))
		if probabilities is None:
			if len(values) != 1:
				raise ValueError("probabilities are required when sampling multiple coupling values.")
			probabilities = np.ones(1)
		probabilities = np.atleast_1d(np.asarray(probabilities,dtype=float))
		if len(values) != len(probabilities) or not np.isclose(np.sum(probabilities),1.):
			raise ValueError("Coupling values and probabilities must have equal length and probabilities must sum to one.")
		for i,j in self.edges(shell):
			Jij = self.rng.choice(values,p=probabilities)
			self.J_matrix[i,j] = self.J_matrix[j,i] = Jij
		self.coupling_specs.append({
			"shell":str(shell),
			"distribution":"choice",
			"values":values.astype(float).tolist(),
			"probabilities":probabilities.astype(float).tolist(),
		})
		self._invalidate_coloring()
		return self.check_symmetric()

	def set_nn_J(self,Jnn,p=1.):
		self.Jnn,self.pnn = Jnn,p
		result = self.set_shell_J("nn",Jnn,values=[Jnn,-Jnn],probabilities=[p,1.-p])
		self.coupling_specs[-1]["model"] = "nn_binary_sign"
		return result

	def set_nn_Gaussian(self,Javg,Jstd):
		self._ensure_writeable_couplings()
		self.Jnn_avg,self.Jnn_std = Javg,Jstd
		for i,j in self.edges("nn"):
			Jij = self.rng.normal(Javg,Jstd)
			self.J_matrix[i,j] = self.J_matrix[j,i] = Jij
		self.coupling_specs.append({
			"shell":"nn",
			"distribution":"normal",
			"mean":float(Javg),
			"std":float(Jstd),
		})
		self._invalidate_coloring()
		return self.check_symmetric()

	def set_nnn_J(self,Jnnn,p=1.):
		self.Jnnn,self.pnnn = Jnnn,p
		result = self.set_shell_J("nnn",Jnnn,values=[Jnnn,0.],probabilities=[p,1.-p])
		self.coupling_specs[-1]["model"] = "nnn_binary_dilution"
		return result

	def _ensure_writeable_couplings(self):
		"""Convert compressed couplings back to efficient assignment form when needed."""
		if not hasattr(self,"coupling_specs"):
			self.coupling_specs = []
		if not isspmatrix_lil(self.J_matrix):
			self.J_matrix = lil_matrix(self.J_matrix)

	def compress_to_csr(self):
		"""Finalize the coupling matrix for memory-efficient arithmetic and storage."""
		if not isinstance(self.J_matrix,csr_matrix):
			self.J_matrix = csr_matrix(self.J_matrix)
		self.J_matrix.eliminate_zeros()
		return self.J_matrix

	def couplings_to_site(self,sites,target):
		"""Return dense couplings from ``sites`` to one target in the same order."""
		sites = np.asarray(sites,dtype=int)
		values = self.J_matrix[sites,int(target)]
		return values.toarray().ravel() if issparse(values) else np.asarray(values).ravel()

	def to_spec(self):
		"""Return compact metadata sufficient to deterministically rebuild this lattice."""
		if self.seed is None:
			raise ValueError("A coupling seed is required to serialize a reconstructable lattice.")
		return {
			"format_version":1,
			"module_version":__version__,
			"geometry":self.geometry,
			"Lx":self.Lx,
			"Ly":self.Ly,
			"periodic":self.periodic,
			"origin_site":self.origin_site,
			"coupling_seed":int(self.seed),
			"couplings":[dict(spec) for spec in self.coupling_specs],
		}

	def edges(self,shell="partners",unique=True):
		if shell == "partners":
			neighbors = self.partners
		else:
			neighbors = self._require_shell(shell)
		edges = []
		for i in self.sites:
			for j in neighbors[i]:
				### unique=True implements one representative i<j per undirected bond.
				if not unique or i < j:
					edges.append((int(i),int(j)))
		return edges

	def edge_crosses_boundary(self,i,j):
		"""Return whether the displayed central-cell edge uses periodic wrapping."""
		if not self.periodic:
			return False
		delta = np.abs(self.cell_indices[int(i)]-self.cell_indices[int(j)])
		return bool(delta[0] > self.Lx/2. or delta[1] > self.Ly/2.)

	def _validate_shell_storage(self):
		if not isinstance(self.neighbor_shells,dict) or not self.neighbor_shells:
			raise ValueError("neighbor_shells must be a non-empty dictionary.")
		for name,shell in self.neighbor_shells.items():
			if len(shell) != self.N:
				raise ValueError(f"Shell '{name}' must contain one neighbor list per site.")
		if "nn" not in self.neighbor_shells:
			raise ValueError(f"Geometry '{self.geometry}' must define an 'nn' shell.")

	def check_symmetric(self):
		difference = self.J_matrix-self.J_matrix.T
		if issparse(difference):
			max_asymmetry = np.max(np.abs(difference.data)) if difference.nnz else 0.
		else:
			max_asymmetry = np.max(np.abs(difference))
		if max_asymmetry > self.symmetry_tol:
			raise ValueError(f"Coupling matrix is not symmetric: max asymmetry = {max_asymmetry}")
		return True

	def validate_topology(self):
		for shell_name,neighbors in self.neighbor_shells.items():
			for i in self.sites:
				if len(neighbors[i]) != len(set(neighbors[i])):
					raise ValueError(f"Duplicate '{shell_name}' neighbor at site {i}.")
				for j in neighbors[i]:
					if j < 0 or j >= self.N:
						raise ValueError(f"Site {j} in shell '{shell_name}' lies outside the lattice.")
					if i not in neighbors[j]:
						raise ValueError(f"Asymmetric '{shell_name}' adjacency between {i} and {j}.")
		return True

	def interaction_colors(self,force_recompute=False):
		if self._interaction_colors is not None and not force_recompute:
			return self._interaction_colors.copy()
		adjacency = [set() for _ in self.sites]
		for i in self.sites:
			for j in self.partners[i]:
				if j != i:
					adjacency[i].add(int(j))
					adjacency[j].add(int(i))
		site_order = sorted(self.sites,key=lambda i: len(adjacency[i]),reverse=True)
		colors = -np.ones(self.N,dtype=int)
		for i in site_order:
			used = {colors[j] for j in adjacency[i] if colors[j] >= 0}
			color = 0
			while color in used:
				color += 1
			colors[i] = color
		self._interaction_colors = colors
		self._interaction_color_classes = [np.where(colors == c)[0] for c in range(np.max(colors)+1)]
		return colors.copy()

	def interaction_color_classes(self,force_recompute=False):
		if self._interaction_color_classes is None or force_recompute:
			self.interaction_colors(force_recompute=force_recompute)
		return [sites.copy() for sites in self._interaction_color_classes]

	def check_interaction_coloring(self):
		colors = self.interaction_colors()
		for i,j in self.edges("partners"):
			if colors[i] == colors[j]:
				raise ValueError(f"Sites {i} and {j} share color {colors[i]}.")
		return True

	### !!! Codex: why is there a Neel mask in the general lattice class but no other order parameters? I think Neel mask should be moved to each child class, to maintain consistency. Only a purely ferromagnetic order-parameter/mask can be defined regardless of lattice -- and by the way should be, since it is missing now.
	def neel_mask(self):
		names,masks = self.order_parameter_masks()
		if "neel" not in names:
			raise ValueError(f"Geometry '{self.geometry}' does not define a Neel order parameter.")
		return masks[names.index("neel")].copy()

	def stripe_masks(self):
		"""Return every geometry-native stripe/stripy mask in declared order."""
		names,masks = self.order_parameter_masks()
		indices = [
			i for i,name in enumerate(names)
			if name.startswith("stripe_") or name.startswith("stripy_")
		]
		if not indices:
			raise ValueError(f"Geometry '{self.geometry}' does not define stripe order parameters.")
		return tuple(masks[i].copy() for i in indices)

	def magnetic_field_mask_zz(self,distances,probe_xy="centroid"):
		"""Return the Ising dipolar kernel, centered on the lattice by default."""
		dxy,distances = self._probe_displacements(distances,probe_xy)
		rho2 = np.sum(dxy**2,axis=1)
		return (2.*distances[None,:]**2-rho2[:,None])/(rho2[:,None]+distances[None,:]**2)**2.5

	def magnetic_field_mask_tensor(self,distances,probe_xy="centroid"):
		"""Return the Heisenberg dipolar tensor, centered on the lattice by default."""
		dxy,distances = self._probe_displacements(distances,probe_xy)
		rvec = np.stack(np.broadcast_arrays(dxy[:,0,None],dxy[:,1,None],distances[None,:]))
		r2 = np.sum(rvec**2,axis=0)
		kernel = 3.*np.einsum("and,bnd->abnd",rvec,rvec)/r2[None,None,...]**2.5
		for a in range(3):
			kernel[a,a] -= 1./r2**1.5
		return kernel

	def _probe_displacements(self,distances,probe_xy):
		distances = np.atleast_1d(np.asarray(distances,dtype=float))
		if probe_xy is None or (isinstance(probe_xy,str) and probe_xy.lower() == "centroid"):
			probe_xy = self.site_centroid
		elif isinstance(probe_xy,str):
			raise ValueError("A string probe_xy setting must be 'centroid'.")
		probe_xy = np.asarray(probe_xy,dtype=float)
		if probe_xy.shape != (2,):
			raise ValueError("probe_xy must be 'centroid' or contain two Cartesian coordinates.")
		### positions and probe_xy share the explicit frame in which origin_site is (0, 0).
		dxy = probe_xy[None,:]-self.positions
		return dxy,distances


class SquareLattice(Lattice):
	geometry = "square"
	nbasis = 1
	primitive_vectors = np.array([[1.,0.],[0.,1.]])
	basis_vectors = np.array([[0.,0.]])

	def _build_neighbor_shells(self):
		nn = self._empty_shell()
		nnn = self._empty_shell()
		for i in self.sites:
			x,y = self.cell_indices[i]
			nn[i] = self._indices_from_specs(i,[(x+1,y,0),(x-1,y,0),(x,y+1,0),(x,y-1,0)])
			nnn[i] = self._indices_from_specs(i,[(x+1,y+1,0),(x+1,y-1,0),(x-1,y+1,0),(x-1,y-1,0)])
		return {"nn":nn,"nnn":nnn}

	### !!! Codex: I don't understand the names indexing. Consider instead returning a list of names, rather than a tuple. What advantage does a tuple confer? It would be good to have something where names can be iterated over as dictionary keys, for example
	def order_parameter_masks(self):
		x,y = self.cell_indices.T
		names = ("neel","stripe_x","stripe_y")
		masks = np.stack([(-1.)**(x+y),(-1.)**x,(-1.)**y])
		return names,masks.astype(float)

class HoneycombLattice(Lattice):
	geometry = "honeycomb"
	nbasis = 2
	primitive_vectors = np.array([[np.sqrt(3.),0.],[np.sqrt(3.)/2.,1.5]]) ### !!! Codex: is this in units where the nn bond lengths are 1? 
	basis_vectors = np.array([[0.,0.],[0.,1.]])

	def _build_neighbor_shells(self):
		nn = self._empty_shell()
		nnn = self._empty_shell()
		nnn_shifts = ((1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1))
		for i in self.sites:
			x,y = self.cell_indices[i]
			sub = self.sublattice[i]
			if sub == 0:
				nn_specs = [(x,y,1),(x,y-1,1),(x+1,y-1,1)]
			else:
				nn_specs = [(x,y,0),(x,y+1,0),(x-1,y+1,0)]
			nn[i] = self._indices_from_specs(i,nn_specs)
			nnn[i] = self._indices_from_specs(i,[(x+dx,y+dy,sub) for dx,dy in nnn_shifts])
		return {"nn":nn,"nnn":nnn}

	### !!! Codex: please add a bit more explaination and context here about the various order parameter masks and how they are constructed.
	### !!! Codex: also please consider describing how they are arranged in terms of +/- when considering a fundamental hexagon of the honeycomb lattice going around, say counterclockwise. 
	### !!! Codex: so, e.g. Neel is (+,-,+,-,+,-) whereas zig-zags can be (+,+,+,-,-,-) and stripy (+,-,-,+,-,-) 
	def order_parameter_masks(self):
		x,y = self.cell_indices.T
		cell_parities = np.stack([(-1.)**x,(-1.)**y,(-1.)**(x+y)])
		sub = np.where(self.sublattice == 0,1.,-1.)
		names = (
			"neel","zigzag_0","zigzag_1","zigzag_2",
			"stripy_0","stripy_1","stripy_2",
		)
		masks = np.concatenate([sub[None,:],cell_parities,cell_parities*sub[None,:]])
		return names,masks.astype(float)


_GEOMETRY_ALIASES = {
	"square":"square","sq":"square",
	"honeycomb":"honeycomb","hc":"honeycomb","hex":"honeycomb","hexagonal":"honeycomb",
}


def canonical_geometry(geometry):
	key = str(geometry).strip().lower()
	if key not in _GEOMETRY_ALIASES:
		allowed = ", ".join(sorted(_GEOMETRY_ALIASES))
		raise ValueError(f"Unknown lattice geometry '{geometry}'. Allowed names and aliases: {allowed}.")
	return _GEOMETRY_ALIASES[key]


def make_lattice(geometry,Lx,Ly=None,periodic=True,origin_site=0):
	geometry = canonical_geometry(geometry)
	classes = {"square":SquareLattice,"honeycomb":HoneycombLattice}
	return classes[geometry](Lx,Ly=Ly,periodic=periodic,origin_site=origin_site)


def square_lattice(Lx,Ly=None,periodic=True,origin_site=0):
	return SquareLattice(Lx,Ly=Ly,periodic=periodic,origin_site=origin_site)


def honeycomb_lattice(Lx,Ly=None,periodic=True,origin_site=0):
	return HoneycombLattice(Lx,Ly=Ly,periodic=periodic,origin_site=origin_site)


def lattice_from_spec(spec):
	"""Rebuild a lattice and its quenched couplings from compact saved metadata."""
	if not isinstance(spec,dict) or spec.get("format_version") != 1:
		raise ValueError("Unsupported lattice specification.")
	latt = make_lattice(
		spec["geometry"],spec["Lx"],Ly=spec["Ly"],
		periodic=spec.get("periodic",True),origin_site=spec.get("origin_site",0),
	)
	latt.set_seed(int(spec["coupling_seed"]))
	for coupling in spec.get("couplings",[]):
		distribution = coupling.get("distribution")
		model = coupling.get("model")
		if model == "nn_binary_sign":
			latt.set_nn_J(coupling["values"][0],coupling["probabilities"][0])
		elif model == "nnn_binary_dilution":
			latt.set_nnn_J(coupling["values"][0],coupling["probabilities"][0])
		elif distribution == "choice":
			latt.set_shell_J(
				coupling["shell"],coupling["values"][0],
				values=coupling["values"],probabilities=coupling["probabilities"],
			)
		elif distribution == "normal" and coupling.get("shell") == "nn":
			latt.set_nn_Gaussian(coupling["mean"],coupling["std"])
		else:
			raise ValueError(f"Unsupported coupling specification: {coupling}.")
	latt.compress_to_csr()
	return latt


### Historical constructor: lattice(L) remains a square lattice.
lattice = SquareLattice
