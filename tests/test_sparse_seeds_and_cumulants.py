import pickle

import numpy as np
import pytest
from scipy.sparse import isspmatrix_csr,isspmatrix_lil

import glauber_dynamics as gd
import heisenberg_dynamics as hd
import lattice_methods as lm
import noise_methods as nm
from simulation_utils import resolve_replica_seeds


@pytest.mark.parametrize("geometry",["square","honeycomb"])
def test_sparse_lattice_round_trip(geometry):
	lattice = lm.make_lattice(geometry,4)
	assert isspmatrix_lil(lattice.J_matrix)
	lattice.set_seed(17)
	lattice.set_nn_J(1.,1.)
	lattice.set_nnn_J(-0.5,0.4)
	expected = lattice.J_matrix.toarray()
	spec = lattice.to_spec()
	lattice.compress_to_csr()
	restored = lm.lattice_from_spec(spec)
	assert isspmatrix_csr(lattice.J_matrix)
	assert lattice.J_matrix.nnz < lattice.N**2
	np.testing.assert_array_equal(restored.J_matrix.toarray(),expected)
	assert restored.Jnnn == pytest.approx(-0.5)
	assert restored.pnnn == pytest.approx(0.4)


def test_legacy_dense_couplings_remain_readable_and_convertible():
	lattice = lm.square_lattice(3)
	lattice.J_matrix = np.zeros((lattice.N,lattice.N))
	assert lattice.check_symmetric()
	lattice.set_seed(2)
	lattice.set_nn_J(1.,1.)
	lattice.compress_to_csr()
	assert isspmatrix_csr(lattice.J_matrix)


def test_constructed_seeds_are_deterministic_complete_and_independent():
	first = resolve_replica_seeds(12,3,construct_seeds=True)
	second = resolve_replica_seeds(12,3,construct_seeds=True)
	other_replica = resolve_replica_seeds(12,4,construct_seeds=True)
	assert first == second
	assert first[0] is not None and first[1] is not None
	assert first[0] != first[1]
	assert first != other_replica
	assert resolve_replica_seeds(12,3,5,7,construct_seeds=False) == (5,7)


def test_batch_result_uses_reconstructable_lattice_metadata(tmp_path):
	output = tmp_path/"result.pkl"
	gd.run_sims(
		str(output),3,-0.5,0.5,7,3,[1.0],[2.0],2,
		initial_seed=11,dynamics_seed=13,construct_seeds=True,
		geometry="honeycomb",use_color_updates=True,
	)
	with output.open("rb") as stream:
		result = pickle.load(stream)
	assert result["format_version"] == 3
	assert "lattice" not in result
	assert result["module_versions"] == {
		"lattice_methods":lm.__version__,
		"glauber_dynamics":gd.__version__,
	}
	assert result["dynamics_parameters"]["use_color_updates"] is True
	assert result["seeds"]["initial"] != result["seeds"]["dynamics"]
	restored = lm.lattice_from_spec(result["lattice_spec"])
	assert restored.geometry == "honeycomb"
	assert isspmatrix_csr(restored.J_matrix)


def test_geometry_native_stripe_masks():
	square = lm.square_lattice(4)
	assert len(square.stripe_masks()) == 2

	honeycomb = lm.honeycomb_lattice(4)
	names,masks = honeycomb.order_parameter_masks()
	cycle = [0,1,8,3,2,27]
	patterns = {name:tuple(mask[cycle].astype(int)) for name,mask in zip(names,masks)}
	assert patterns["neel"] == (1,-1,1,-1,1,-1)
	assert (1,1,1,-1,-1,-1) in {
		patterns[name] for name in names if name.startswith("zigzag_")
	}
	assert (1,-1,-1,1,-1,-1) in {
		patterns[name] for name in names if name.startswith("stripy_")
	}
	assert len(honeycomb.stripe_masks()) == 3


@pytest.mark.parametrize("geometry",["square","honeycomb"])
def test_probe_defaults_to_centroid_independent_of_coordinate_origin(geometry):
	first = lm.make_lattice(geometry,4,origin_site=0)
	second = lm.make_lattice(geometry,4,origin_site=first.N//2)
	first_displacements,distances = first._probe_displacements([2.],"centroid")
	second_displacements,_ = second._probe_displacements([2.],None)

	np.testing.assert_allclose(np.mean(first_displacements,axis=0),[0.,0.],atol=1.e-14)
	np.testing.assert_allclose(first_displacements,second_displacements,atol=1.e-14)
	np.testing.assert_allclose(
		first.magnetic_field_mask_zz(distances),
		second.magnetic_field_mask_zz(distances),
		atol=1.e-14,
	)
	assert not np.allclose(
		first.magnetic_field_mask_zz(distances),
		first.magnetic_field_mask_zz(distances,probe_xy=(0.,0.)),
	)


def test_positive_heisenberg_coupling_is_antiferromagnetic_and_damped():
	lattice = lm.square_lattice(4)
	lattice.set_seed(3)
	lattice.set_nn_J(1.,1.)
	lattice.compress_to_csr()
	simulation = hd.dynamics(lattice,nsteps=1000,temp=0.,gilbert=1.,dt=5.e-3)

	uniform = np.zeros((3,lattice.N))
	uniform[2] = 1.
	neel = uniform*lattice.neel_mask()[None,:]
	ground_energy = simulation.energy(neel)
	assert ground_energy < simulation.energy(uniform)

	simulation.set_seed(5)
	result = simulation.run()
	assert result["energy"][-1] < result["energy"][0]
	assert result["energy"][-1] == pytest.approx(ground_energy,abs=5.e-5)
	assert np.linalg.norm(simulation.neel_order()) > 0.99
	assert result["module_versions"]["heisenberg_dynamics"] == hd.__version__


def test_hahn_cumulants_pool_one_replica_across_disorder():
	positive = np.zeros((1,1,1,20))
	negative = np.zeros_like(positive)
	### After the 20% chop these entries produce equal X/Y echoes with signs +/-.
	positive[...,4] = positive[...,6] = 1.
	negative[...,4] = negative[...,6] = -1.

	times,cumulants = nm.annealed_hahn_cumulants([positive,negative],sample_size=1)
	single = nm.extract_cumulants_Hahn(nm.down_sample(positive,4,1))
	assert times[1] == 1
	assert single[2,0,0,1] == pytest.approx(0.)
	assert cumulants[2,0,0,1] == pytest.approx(1.)
	assert cumulants[11,0,0,1] == pytest.approx(-2.)

	processed = nm.process_cumulants_av_MGF([positive,negative],sample_size=1)
	assert processed[1][0,0,1] == pytest.approx(1.)
	assert processed[2][0,0,1] == pytest.approx(-2.)


def test_stripe_postprocessing_uses_replica_first_axis():
	mags = [np.zeros((1,2,3))]
	neels = [np.zeros((1,2,3))]
	stripes = [np.zeros((1,2,2,3))]
	stripes[0][0,0,:,-1] = [1.,2.]
	stripes[0][0,1,:,-1] = [3.,4.]
	_,_,stripe_order = nm.order_parameters(mags,neels,stripes)
	np.testing.assert_array_equal(stripe_order,[4.,6.])
