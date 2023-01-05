import sys
sys.path.insert(1, 'graphnn/')

from graphnn.data import TransformRowToGraphXyz
import numpy as np
import torch

######################################################################################
# Code modified from the graphh package by Peter Bjørn Jørgensen (pbjo@dtu.dk) et al.
######################################################################################
class TransformRowToGraphXyzWithZpve(TransformRowToGraphXyz):
    def __init__(self, cutoff=5.0, energy_property="energy", forces_property="forces", U0_to_E = False):
        self.cutoff = cutoff
        self.energy_property = energy_property
        self.forces_property = forces_property
        self.U0_to_E = U0_to_E

    def __call__(self, row):
        atoms = row.toatoms()

        if np.any(atoms.get_pbc()):
            atoms.wrap()  # Make sure all atoms are inside unit cell
            edges, edges_displacement = self.get_edges_neighborlist(atoms)
        else:
            edges, edges_displacement = self.get_edges_simple(atoms)

        # Extract energy and forces if they exists
        energy = np.array([0.0])
        try:
            energy = np.array([getattr(row, self.energy_property)])
        except AttributeError:
            pass
        try:
            energy = np.copy([np.squeeze(row.data[self.energy_property])])
        except (KeyError, AttributeError):
            pass
        forces = np.zeros((len(atoms), 3))
        try:
            forces = np.copy(getattr(row, self.forces_property))
        except AttributeError:
            pass
        try:
            forces = np.copy(row.data[self.forces_property])
        except (KeyError, AttributeError):
            pass
            
        # Extract zpve, if it exists
        zpve = np.array([0.0])
        try:
            zpve = np.copy(getattr(row, "zpve"))
        except AttributeError:
            pass
        try:
            zpve = np.copy(row.data["zpve"])
        except (KeyError, AttributeError):
            pass

        # Extract index, if it exists.
        index = np.array([-1])
        try:
            index = np.copy(getattr(row, "index"))
        except AttributeError:
            pass
        try:
            index = np.copy(row.data["index"])
        except (KeyError, AttributeError):
            pass

        default_type = torch.get_default_dtype()

        if self.U0_to_E:
            energy = energy - zpve

        # pylint: disable=E1102
        graph_data = {
            "nodes": torch.tensor(atoms.get_atomic_numbers()),
            "nodes_xyz": torch.tensor(atoms.get_positions(), dtype=default_type),
            "num_nodes": torch.tensor(len(atoms.get_atomic_numbers())),
            "edges": torch.tensor(edges),
            "edges_displacement": torch.tensor(edges_displacement, dtype=default_type),
            "cell": torch.tensor(np.array(atoms.get_cell()), dtype=default_type),
            "num_edges": torch.tensor(edges.shape[0]),
            "energy": torch.tensor(energy, dtype=default_type),
            "forces": torch.tensor(forces, dtype=default_type),
            "zpve": torch.tensor(zpve, dtype=default_type),
            "index": torch.tensor(index, dtype=torch.int32)
        }

        return graph_data