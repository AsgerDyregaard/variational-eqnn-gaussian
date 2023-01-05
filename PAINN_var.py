import math

import torch
from torch import nn
from torch.distributions.normal import Normal

import sys
import os
sys.path.insert(1, os.path.join(os.path.abspath(os.path.dirname(__file__)),'graphnn'))
from graphnn import layer
import layer_variational as lv
import var_layer as varl

# For debugging purposes
import importlib
importlib.reload(layer)
importlib.reload(lv)
importlib.reload(varl)
#.

######################################################################################
# Code modified from the graphh package by Peter Bjørn Jørgensen (pbjo@dtu.dk) et al.
######################################################################################
class PainnModel_variational(nn.Module):
    """PainnModel with forces."""

    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        target_mean=[0.0],
        target_stddev=[1.0],
        normalize_atomwise=True,
        direct_force_output=False,
        standard_deviance=1e-8,
        **kwargs,
    ):
        """
        Args:
            num_interactions (int): Number of interaction layers
            hidden_state_size (int): Size of hidden node states
            cutoff (float): Atomic interaction cutoff distance [Å]
            target_mean ([float]): Target normalisation constant
            target_stddev ([float]): Target normalisation constant
            normalize_atomwise (bool): Use atomwise normalisation
            direct_force_output (bool): Compute forces directly instead of using gradient
        """
        super().__init__(**kwargs)
        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.cutoff = cutoff
        self.distance_embedding_size = 20
        self.standard_deviance = standard_deviance

        num_embeddings = 119  # atomic numbers + 1
        edge_size = self.distance_embedding_size

        # Setup atom embeddings
        self.atom_embeddings = nn.Embedding(num_embeddings, hidden_state_size)

        # Setup interaction networks
        self.interactions = nn.ModuleList(
            [
                lv.PaiNNInteraction(hidden_state_size, edge_size, self.cutoff)
                for _ in range(num_interactions)
            ]
        )
        self.scalar_vector_update = nn.ModuleList(
            [lv.PaiNNUpdate(hidden_state_size) for _ in range(num_interactions)]
        )

        # Setup readout function
        self.readout_mlp = nn.Sequential(
            varl.linear_var(hidden_state_size, hidden_state_size),
            nn.SiLU(),
            varl.linear_var(hidden_state_size, 1),
        )

        # Normalisation constants
        self.normalize_atomwise = torch.nn.Parameter(
            torch.tensor(normalize_atomwise), requires_grad=False
        )
        self.normalize_stddev = torch.nn.Parameter(
            torch.as_tensor(target_stddev), requires_grad=False
        )
        self.normalize_mean = torch.nn.Parameter(
            torch.as_tensor(target_mean), requires_grad=False
        )

        # Direct force output
        self.direct_force_output = direct_force_output
        if self.direct_force_output:
            self.force_readout_linear = varl.linear_var(hidden_state_size, 1, bias=False)
    
    def get_divergence(self):
        kl = 0
        for int_layer, update_layer in zip(
            self.interactions, self.scalar_vector_update
        ):
            kl += int_layer.filter_layer.get_divergence()
            kl += int_layer.scalar_message_mlp[0].get_divergence()
            kl += int_layer.scalar_message_mlp[2].get_divergence()

            kl += update_layer.linearU.get_divergence()
            kl += update_layer.linearV.get_divergence()
            kl += update_layer.combined_mlp[0].get_divergence()
            kl += update_layer.combined_mlp[2].get_divergence()

        kl += self.readout_mlp[0].get_divergence()
        kl += self.readout_mlp[2].get_divergence()

        if self.direct_force_output:
            kl += self.force_readout_linear.get_divergence()
        
        return kl
    
    def sample_to_batch(self, input_dict, n_samples):
        # Turns a single sample into a batch
        sample_batch = {}
        for key in input_dict.keys():
            val = input_dict[key]
            if len(val.shape) == 3:
                sample_batch[key] = val.repeat([n_samples,1,1])
            
            if len(val.shape) == 2:
                sample_batch[key] = val.repeat([n_samples,1])

            if len(val.shape) == 1:
                sample_batch[key] = val.repeat([n_samples,])
        
        return sample_batch
    
    def sample(self, input_dict, compute_forces=True, compute_stress=True, n_samples=100):
        # Samples n_samples energy estimates for a single given input.

        sample_batch = self.sample_to_batch(input_dict, n_samples)
        return self.forward(sample_batch, compute_forces, compute_stress)["energy"].detach()

    def forward(self, input_dict, compute_forces=True, compute_stress=True):
        """
        Args:
            input_dict (dict): Input dictionary of tensors with keys: nodes,
                               nodes_xyz, num_nodes, edges, edges_displacement, cell,
                               num_edges, targets
        Returns:
            result_dict (dict): Result dictionary with keys:
                                energy, forces, stress
                                Forces and stress are only included if requested (default).
        """
        if compute_forces and not self.direct_force_output:
            input_dict["nodes_xyz"].requires_grad_()
        if compute_stress:
            # Create displacement matrix of zeros and transform cell and atom positions
            displacement = torch.zeros_like(input_dict["cell"], requires_grad=True)
            input_dict["cell"] = input_dict["cell"] + torch.matmul(
                input_dict["cell"], displacement
            )
            input_dict["nodes_xyz"] = input_dict["nodes_xyz"] + torch.matmul(
                input_dict["nodes_xyz"], displacement
            )

        # Unpad and concatenate edges and features into batch (0th) dimension
        edges_displacement = layer.unpad_and_cat(
            input_dict["edges_displacement"], input_dict["num_edges"]
        )
        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["edges"] + edge_offset
        edges = layer.unpad_and_cat(edges, input_dict["num_edges"])

        # Unpad and concatenate all nodes into batch (0th) dimension
        nodes_xyz = layer.unpad_and_cat(
            input_dict["nodes_xyz"], input_dict["num_nodes"]
        )
        nodes_scalar = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        nodes_scalar = self.atom_embeddings(nodes_scalar)
        nodes_vector = torch.zeros(
            (nodes_scalar.shape[0], 3, self.hidden_state_size),
            dtype=nodes_scalar.dtype,
            device=nodes_scalar.device,
        )

        # Compute edge distances
        edges_distance, edges_diff = layer.calc_distance(
            nodes_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_edges"],
            return_diff=True,
        )

        # Expand edge features in Gaussian basis
        edge_state = layer.sinc_expansion(
            edges_distance, [(self.distance_embedding_size, self.cutoff)]
        )
        # Apply interaction layers
        for int_layer, update_layer in zip(
            self.interactions, self.scalar_vector_update
        ):
            nodes_scalar, nodes_vector = int_layer(
                nodes_scalar,
                nodes_vector,
                edge_state,
                edges_diff,
                edges_distance,
                edges,
            )
            nodes_scalar, nodes_vector = update_layer(nodes_scalar, nodes_vector)

        # Apply readout function
        nodes_scalar = self.readout_mlp(nodes_scalar)

        # Obtain graph level output
        graph_output = layer.sum_splits(nodes_scalar, input_dict["num_nodes"])

        # Apply (de-)normalization
        normalizer = self.normalize_stddev.unsqueeze(0)
        graph_output = graph_output * normalizer
        mean_shift = self.normalize_mean.unsqueeze(0)
        if self.normalize_atomwise:
            mean_shift = mean_shift * input_dict["num_nodes"].unsqueeze(1)
        graph_output = graph_output + mean_shift

        result_dict = {"energy": graph_output}

        # Compute forces
        if compute_forces:
            if self.direct_force_output:
                forces = self.force_readout_linear(nodes_vector)
                forces = torch.squeeze(forces, 2)

                forces_reshaped = layer.pad_and_stack(
                    torch.split(
                        forces,
                        list(input_dict["num_nodes"].detach().cpu().numpy()),
                        dim=0,
                    )
                )
                result_dict["forces"] = forces_reshaped
            else:
                dE_dxyz = torch.autograd.grad(
                    graph_output,
                    input_dict["nodes_xyz"],
                    grad_outputs=torch.ones_like(graph_output),
                    retain_graph=True,
                    create_graph=True,
                )[0]
                forces = -dE_dxyz
                result_dict["forces"] = forces
        # Compute stress
        if compute_stress:
            stress = torch.autograd.grad(
                graph_output,
                displacement,
                grad_outputs=torch.ones_like(graph_output),
                retain_graph=True,
                create_graph=True,
            )[0]
            # Compute cell volume
            cell = input_dict["cell"]
            volume = torch.sum(
                cell[:, 0, :] * torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
                dim=1,
                keepdim=True,
            )[..., None]
            stress = stress / volume
            result_dict["stress"] = stress

        return result_dict
