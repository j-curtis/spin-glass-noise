### Shared helpers for reproducible batch simulations and compact saved results

import numpy as np


def resolve_replica_seeds(
	J_seed,
	replica,
	initial_seed=None,
	dynamics_seed=None,
	construct_seeds=False,
):
	"""Resolve independent initial-state and dynamics seeds for one replica.

	When ``construct_seeds`` is false, explicitly supplied seeds are preserved.
	When it is true, both seeds are deterministically derived from the coupling
	seed, replica index, optional user seed, and a distinct stream identifier.
	"""
	J_seed = int(J_seed)
	replica = int(replica)
	if J_seed < 0 or replica < 0:
		raise ValueError("J_seed and replica must be non-negative integers.")

	if not construct_seeds:
		return (
			None if initial_seed is None else int(initial_seed),
			None if dynamics_seed is None else int(dynamics_seed),
		)

	def derive(optional_seed,stream_id):
		base_seed = 0 if optional_seed is None else int(optional_seed)
		if base_seed < 0:
			raise ValueError("Base seeds must be non-negative integers.")
		sequence = np.random.SeedSequence([J_seed,replica,base_seed,stream_id])
		return int(sequence.generate_state(1,dtype=np.uint64)[0])

	return derive(initial_seed,0),derive(dynamics_seed,1)


def compact_result_for_storage(result,J_seed,replica,initial_seed,dynamics_seed,construct_seeds):
	"""Replace a lattice object with reconstructable metadata for serialized output."""
	stored = dict(result)
	lattice = stored.pop("lattice")
	stored["format_version"] = 3
	stored["lattice_spec"] = lattice.to_spec()
	stored["seeds"] = {
		"coupling":int(J_seed),
		"replica":int(replica),
		"initial":initial_seed,
		"dynamics":dynamics_seed,
		"constructed":bool(construct_seeds),
	}
	return stored
