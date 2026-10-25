import copy
import itertools
import os
import random
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import biotite.structure as struc
import numpy as np
import torch
from cifutils.constants import CRYSTALLIZATION_AIDS, ELEMENT_NAME_TO_ATOMIC_NUMBER
from cifutils.enums import ChainType
from cifutils.transforms.categories import initialize_chain_info_from_category
from cifutils.utils.io_utils import read_any, to_cif_file
from cifutils.utils.non_rcsb import initialize_chain_info_from_atom_array

from .residue_mapping import (
    BACKBONE_ATOMS,
    PROTEIN_ALL_ATOMS,
    restype_1to3,
    restype_3to1,
    restype_INTtoSTR,
    restype_STRtoINT,
)


class BaseTransform(metaclass=ABCMeta):
    """Base class for all transformations."""

    def __call__(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:

        return self.transform(results)

    @abstractmethod
    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """The transform function. All subclass of BaseTransform should
        override this method.

        This function takes the result dict as the input, and can add new
        items to the dict or modify existing items in the dict. And the result
        dict will be returned in the end, which allows to concate multiple
        transforms into a pipeline.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """


class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict, callable], optional): Sequence of transform
            object to be composed.
    """

    def __init__(self, transforms: Optional[Sequence[Union[dict, Callable]]]):
        self.transforms: List[Callable] = []

        if transforms is None:
            transforms = []

        for transform in transforms:
            if callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(
                    f"transform must be a callable object or dict, " f"but got {type(transform)}"
                )

    def __call__(self, data: dict) -> Optional[dict]:
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            # The transform will return None when it failed to load images or
            # cannot find suitable augmentation parameters to augment the data.
            # Here we simply return None if the transform returns None and the
            # dataset will handle it by randomly selecting another data sample.
            if data is None:
                return None
        return data

    def __repr__(self):
        """Print ``self.transforms`` in sequence.

        Returns:
            str: Formatted string.
        """
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class LoadFile(BaseTransform):
    """A wrapper around the AI/cifutils `read_any` function that reads in a CIF or PDB file.

    Args:
        - results with keys:
            - cif_path (str): The file path where the CIF formatted structure will be saved.

    Returns:
        - results with keys:
            - cif_path (str): The file path where the CIF formatted structure will be saved.
            - assembly_id (str): The assembly id.
            - file_obj (pdbx.CIFFile | biotite_pdb.PDBFile | pdbx.BinaryCIFFile): The file object.
    """

    def transform(self, results):
        cif_path = results["cif_path"]
        file_obj = read_any(cif_path)
        results["file_obj"] = file_obj
        return results


class GetStructure(BaseTransform):
    """Directly load the structure, suitable for PDB file"""

    def __init__(self, random_select_model=False):
        """
        Args:
            random_select_model: whether to randomly select a model, will return the first model if False
        """
        self.random_select_model = random_select_model

    def _choose_model(self, atom_array):
        if len(atom_array.shape) > 1:
            if self.random_select_model:
                i = random.randint(0, atom_array.shape[0] - 1)
                atom_array = atom_array[i]
            else:
                atom_array = atom_array[0]
        return atom_array

    def transform(self, results):
        cif_path = results["cif_path"]
        file_obj = results["file_obj"]

        filename = cif_path.split("/")[-1]
        if ".cif" in filename:
            atom_array = struc.io.pdbx.get_structure(
                file_obj,
                extra_fields=["occupancy", "b_factor"],
            )
        elif ".pdb" in filename:
            atom_array = struc.io.pdb.get_structure(
                file_obj, extra_fields=["occupancy", "b_factor"]
            )
        else:
            raise ValueError(f"Unknown file type: {filename}")

        atom_array = self._choose_model(atom_array)

        results["atom_array"] = atom_array
        return results


class GetAssembly(BaseTransform):
    """Get the specific assembly from the structure file

    Args:
        - results with keys:
            - cif_path (str): The file path where the CIF formatted structure will be saved.
            - assembly_id (str): The assembly id.
            - file_obj (pdbx.CIFFile | biotite_pdb.PDBFile | pdbx.BinaryCIFFile): The file object.

    Returns:
        - results with keys:
            - cif_path (str): The file path where the CIF formatted structure will be saved.
            - assembly_id (str): The assembly id.
            - file_obj (pdbx.CIFFile | biotite_pdb.PDBFile | pdbx.BinaryCIFFile): The file object.
            - atom_array (AtomArray): The AtomArray object.
    """

    def transform(self, results):
        cif_path = results["cif_path"]
        assembly_id = results["assembly_id"]
        file_obj = results["file_obj"]

        filename = cif_path.split("/")[-1]
        build_assembly = True
        extra_fields = ["occupancy"]
        if ".cif" in filename:
            if "pdbx_struct_assembly_gen" in file_obj.block and "pdbx_struct_oper_list" in file_obj.block:
                func = partial(struc.io.pdbx.get_assembly, use_author_fields=False)
            else:
                print(
                    "Assembly-building annotations not found in CIF file. This likely indicates a file "
                    "that is not from the RCSB PDB. The AtomArray will be created directly from the `atom_site`. "
                    "If multiple copies of the asymmetric unit are present, unexpected behavior may result. "
                    )
                func = partial(struc.io.pdbx.get_structure, use_author_fields=False)
                build_assembly=False
            # Keep the "is_motif_atom_with_fixed_seq" annotation if present in the CIF file
            atom_site_category = file_obj.block["atom_site"]
            if "is_motif_atom_with_fixed_seq" in atom_site_category:
                extra_fields.append("is_motif_atom_with_fixed_seq")
        elif ".pdb" in filename:
            # Default to just getting the monomer structure if assembly_id is None
            func = struc.io.pdb.get_structure
            build_assembly = False
            if assembly_id is not None:
                func = struc.io.pdb.get_assembly
                build_assembly = True
        else:
            raise ValueError(f"Unknown file type: {filename}")

        if build_assembly:
            atom_array = func(
                file_obj,
                assembly_id,
                extra_fields=extra_fields,
            )
        else:
            atom_array = func(
                file_obj,
                extra_fields=extra_fields,
            )

        if len(atom_array.shape) > 1:
            # assert (
            #     atom_array.shape[0] == 1
            # ), f"Multiple models ({atom_array.shape}) found in {cif_path}"
            if (model_num := atom_array.shape[0]) > 1:
                warnings.warn(f"Multiple ({model_num}) models found in {cif_path}")
            atom_array = atom_array[0]  # TODO: can only take one because of cache

        results["atom_array"] = atom_array
        return results


class ExcludeByResName(BaseTransform):
    """Exclude residues by res_name

    Args:
        - results with keys:
            - atom_array (AtomArray): The AtomArray object.

    Returns:
        - results with keys:
            - atom_array (AtomArray): The AtomArray object.
    """

    def __init__(self, exclude_res_list: List[str]):
        exclude_by_default = CRYSTALLIZATION_AIDS + ["HOH", "NA", "CL", "K", "BR", "UNX"]
        self.exclude_res_list = list(set(exclude_res_list + exclude_by_default))

    def transform(self, results):
        atom_array = results["atom_array"]
        mask = np.isin(atom_array.res_name, self.exclude_res_list, invert=True)
        results["atom_array"] = atom_array[mask]
        return results


class GetChainInfo(BaseTransform):
    """Get chain info from the atom_array"""

    def transform(self, results):

        atom_array = results["atom_array"]
        try:
            chain_info = initialize_chain_info_from_category(
                results["file_obj"].block, atom_array
            )  # TODO: this will never succeed: Annotation category 'label_entity_id' is not existing
        except Exception as e:
            # print(f"Error in getting chain info: {e}")  # mute this info for design people to avoid confusion
            chain_info = initialize_chain_info_from_atom_array(atom_array)
        results["chain_info"] = chain_info
        return results


class SeperateProteinLigand(BaseTransform):
    """Extract ligand from the atom_array

    Args:
        - results with keys:
            - atom_array (AtomArray): The AtomArray object.

    Returns:
        - results with keys:
            - atom_array (AtomArray): The AtomArray object.
            - poly_chain_dict (dict): The dictionary of protein chains.
            - ligand_chain_dict (dict): The dictionary of ligand chains. This comprises all non-protein chains.
            - extra_ligand_chain_dict (dict): The dictionary of extra ligand chains. This refers to atoms on a primarily
                polypeptide chain which are not part of a standard amino acid, e.g. non-canonical or glycan atoms.
            - other_atoms (AtomArray): An AtomArray consisting of atoms in the ligand_chain_dict or the extra_ligand_chain_dict
    """

    def __init__(
        self,
        atoms_to_load="all",
        excluded_elements=['H', 'X'],
        protein_backbone_occ_cutoff=0.8,
        protein_side_chain_occ_cutoff=0.5,
        ligand_occ_cutoff=0.01,
    ):
        assert atoms_to_load in ["all", "backbone"]
        self.atom_list_to_load = atoms_to_load
        self.excluded_elements = set(excluded_elements)
        self.protein_backbone_occ_cutoff = protein_backbone_occ_cutoff
        self.protein_side_chain_occ_cutoff = protein_side_chain_occ_cutoff
        self.ligand_occ_cutoff = ligand_occ_cutoff

    def transform(self, results):
        atom_array = results["atom_array"]

        chain_info = results["chain_info"]

        protein_chain_list = []
        ligand_chain_list = []

        for chain_letter, v in chain_info.items():
            curr_chain_array = atom_array[atom_array.chain_id == chain_letter]
            canonical_aa_mask = struc.filter_canonical_amino_acids(curr_chain_array)
            num_canonical_aa_in_chain = len(np.unique(curr_chain_array[canonical_aa_mask].res_id))

            if v["chain_type"] == ChainType.POLYPEPTIDE_L and num_canonical_aa_in_chain > 4:
                protein_chain_list.append(chain_letter)
            else:
                ligand_chain_list.append(chain_letter)

        poly_chain_dict = {}
        # ligand atoms that are in the protein chain (e.g. unstandard amino acids)
        extra_ligand_chain_dict = {}

        if self.atom_list_to_load == "backbone":
            atom_list_to_save = BACKBONE_ATOMS
        elif self.atom_list_to_load == "all":
            atom_list_to_save = PROTEIN_ALL_ATOMS
        else:
            raise ValueError(f"Unknown atoms_to_load: {self.atom_list_to_load}")

        # Get an atom array consisting of the `extra_ligands`
        protein_chain_mask = np.isin(atom_array.chain_id, protein_chain_list)
        nonstandard_resname_mask = ~np.isin(atom_array.res_name, list(restype_3to1.keys()))
        non_excluded_element_mask =  ~np.isin(atom_array.element, self.excluded_elements)
        sufficient_occupancy_mask = atom_array.occupancy > self.ligand_occ_cutoff
        extra_ligand_mask = protein_chain_mask & nonstandard_resname_mask & non_excluded_element_mask & sufficient_occupancy_mask
        extra_ligand_atom_array = atom_array[extra_ligand_mask]

        # Get an AtomArray consisting of the `ligands`
        ligand_chain_mask = np.isin(atom_array.chain_id, ligand_chain_list)
        ligand_mask = ligand_chain_mask & non_excluded_element_mask & sufficient_occupancy_mask
        ligand_atom_array = atom_array[ligand_mask]

        for chain_id in protein_chain_list:
            extra_ligand_xyz_list = []
            extra_ligand_element_list = []

            atoms = atom_array[atom_array.chain_id == chain_id]
            L = struc.get_residue_count(atoms)
            xyz = np.zeros([L, len(atom_list_to_save), 3], dtype=np.float32)
            occ = np.zeros([L, len(atom_list_to_save)], dtype=np.float32)
            residue_idx = -100 * np.ones([L], dtype=np.int32)
            raw_sequence = L * ["UNK"]

            for c, (res_id, atom_tuple) in enumerate(itertools.groupby(atoms, lambda x: x.res_id)):
                for atom in atom_tuple:
                    if atom.res_name not in restype_3to1:
                        if (
                            atom.element not in self.excluded_elements
                            and atom.occupancy > self.ligand_occ_cutoff
                        ):
                            extra_ligand_xyz_list.append(atom.coord)
                            extra_ligand_element_list.append(np.array(atom.element))
                    else:  # standard amino acid
                        if atom.atom_name in atom_list_to_save:
                            atom_idx = atom_list_to_save.index(atom.atom_name)
                            xyz[c, atom_idx, :] = atom.coord
                            occ[c, atom_idx] = atom.occupancy
                            if atom.atom_name == "CA":
                                raw_sequence[c] = atom.res_name
                                residue_idx[c] = int(atom.res_id)

            if len(extra_ligand_xyz_list) > 0:
                extra_ligand_chain_dict[chain_id] = {
                    "xyz": np.stack(extra_ligand_xyz_list),
                    "element": np.stack(extra_ligand_element_list),
                }

            mask_occ = occ > self.protein_backbone_occ_cutoff
            mask_occ = mask_occ[:, :4].sum(axis=1) == 4  # all backbone atoms

            poly_chain_dict[chain_id] = {
                "xyz": xyz[mask_occ],
                "occ": occ[mask_occ],
                "seq": [res_name for i, res_name in enumerate(raw_sequence) if mask_occ[i]],
                "residue_idx": residue_idx[mask_occ],
            }
            poly_chain_dict[chain_id]["side_chain_mask"] = (
                poly_chain_dict[chain_id]["occ"] > self.protein_side_chain_occ_cutoff
            )

        ligand_chain_dict = {}
        for chain_id in ligand_chain_list:
            xyz_list = []
            element_list = []
            type_list = []
            for atom in atom_array[atom_array.chain_id == chain_id]:
                if atom.element not in self.excluded_elements:
                    if atom.occupancy > self.ligand_occ_cutoff:
                        xyz_list.append(atom.coord)
                        element_list.append(np.array(atom.element))

            if len(xyz_list) > 0:
                ligand_chain_dict[chain_id] = {
                    "xyz": np.stack(xyz_list),
                    "element": np.stack(element_list),
                    "res_type": type_list,
                }

        results["poly_chain_dict"] = poly_chain_dict
        results["ligand_chain_dict"] = ligand_chain_dict
        results["extra_ligand_chain_dict"] = extra_ligand_chain_dict
        results["other_atoms"] = ligand_atom_array + extra_ligand_atom_array

        return results


class BuildMPNNInput(BaseTransform):
    """Build up original MPNN input

    Args:
        - results with keys:
            - atom_array (AtomArray): The AtomArray object.
            - poly_chain_dict (dict): The dictionary of protein chains.
            - ligand_chain_dict (dict): The dictionary of ligand chains.
            - extra_ligand_chain_dict (dict): The dictionary of extra ligand chains.

    Returns:
        - results with keys:
            - atom_array (AtomArray): The AtomArray object.
            - poly_chain_dict (dict): The dictionary of protein chains.
            - ligand_chain_dict (dict): The dictionary of ligand chains.
            - extra_ligand_chain_dict (dict): The dictionary of extra ligand chains.
            - X (np.ndarray): The coordinates of the protein backbone atoms.
            - X_m (np.ndarray): The mask of the protein backbone atoms.
            - Y (np.ndarray): The coordinates of the ligand atoms.
            - Y_t (np.ndarray): The element of the ligand atoms.
            - Y_coords (np.ndarray): The coordinates of the ligand atoms.
            - S (np.ndarray): The sequence of the protein.
            - R_idx (np.ndarray): The residue index of the protein.
            - chain_labels (np.ndarray): The chain labels of the protein.
            - chain_letters (np.ndarray): The chain letters of the protein.
            - ins_codes (np.ndarray): The insertion codes of the protein.
            - mask (np.ndarray): The mask of the protein.
            - Y_m (np.ndarray): The mask of the ligand atoms.
            - mask_XY (np.ndarray): The mask of the backbone and ligand atoms.
    """

    def __init__(self, lig_nn_num=25, lig_cutoff_for_score=8.0):
        """
        Args:
            lig_nn_num: number of nearest ligand atoms (to Cb) to consider
        """
        self.lig_nn_num = lig_nn_num
        self.lig_cutoff_for_score = lig_cutoff_for_score

    def transform(self, results):
        poly_chain_dict = results["poly_chain_dict"]
        ligand_chain_dict = results["ligand_chain_dict"]
        extra_ligand_chain_dict = results["extra_ligand_chain_dict"]

        Cb_list = []
        X_list = []
        side_chain_mask_list = []
        seq_int_list = []
        res_idx_list = []
        chain_labels_list = []
        chain_letters_list = []

        lig_coords = []
        lig_elements = []

        chain_list = np.unique(results["atom_array"].chain_id)
        for chain_idx, chain_id in enumerate(chain_list):
            if chain_id in poly_chain_dict:
                chain_data = poly_chain_dict[chain_id]
                tmp_residue_idx = chain_data["residue_idx"]
                xyz = chain_data["xyz"]  # TODO: do we need to transform the coordinates?
                N = xyz[:, 0, :]
                CA = xyz[:, 1, :]
                C = xyz[:, 2, :]

                b = CA - N
                c = C - CA
                a = np.cross(b, c, axis=-1)
                CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

                Cb_list.append(CB)
                X_list.append(xyz)
                side_chain_mask_list.append(chain_data["side_chain_mask"])

                seq_one_letter = [restype_3to1[AA] for AA in chain_data["seq"]]
                seq_int = [restype_STRtoINT[AA] for AA in seq_one_letter]
                seq_int_list.append(seq_int)
                res_idx_list.append(tmp_residue_idx)
                chain_labels_list.append(chain_idx * np.ones_like(tmp_residue_idx, dtype=np.int32))
                chain_letters_list.extend([chain_id] * len(tmp_residue_idx))

                if chain_id in extra_ligand_chain_dict:
                    xyz = extra_ligand_chain_dict[chain_id]["xyz"]
                    lig_coords.append(xyz)
                    lig_elements.append(extra_ligand_chain_dict[chain_id]["element"])

            elif chain_id in ligand_chain_dict:
                xyz = ligand_chain_dict[chain_id]["xyz"]
                lig_coords.append(xyz)
                lig_elements.append(ligand_chain_dict[chain_id]["element"])

        if len(Cb_list) > 0:
            Cb_stack = np.concatenate(Cb_list, axis=0)
            X_stack = np.concatenate(X_list, axis=0)
            side_chain_mask_stack = np.concatenate(side_chain_mask_list, axis=0)
            seq_int_stack = np.concatenate(seq_int_list, axis=0)
            res_idx_stack = np.concatenate(res_idx_list, axis=0)
            chain_labels_stack = np.concatenate(chain_labels_list, axis=0)
        else:
            Cb_stack = np.zeros([0, 3], dtype=np.float32)
            X_stack = np.zeros([0, 37, 3], dtype=np.float32)
            side_chain_mask_stack = np.zeros([0, 37], dtype=bool)
            seq_int_stack = np.zeros([0], dtype=np.int32)
            res_idx_stack = np.zeros([0], dtype=np.int32)
            chain_labels_stack = np.zeros([0], dtype=np.int32)

        if len(lig_coords) > 0:
            lig_coords = np.concatenate(lig_coords, axis=0)
            lig_elements = np.concatenate(lig_elements, axis=0)
            lig_elements = np.array(
                [ELEMENT_NAME_TO_ATOMIC_NUMBER[i.upper()] for i in lig_elements],
                dtype=np.int64,
            )

            D = np.sum((Cb_stack[:, None, :] - lig_coords[None, :, :]) ** 2, axis=-1)
            closest_idx = np.argsort(D, axis=-1)[:, : self.lig_nn_num]
            ligand_xyz_closest = np.take_along_axis(
                np.repeat(lig_coords[None, :, :], Cb_stack.shape[0], 0),
                np.repeat(closest_idx[:, :, None], 3, -1),
                axis=1,
            )
            ligand_element_closest = np.take_along_axis(
                np.repeat(lig_elements[None, :], Cb_stack.shape[0], 0), closest_idx, axis=1
            )
            D_XY = np.sqrt(np.sum((Cb_stack - ligand_xyz_closest[:, 0, :]) ** 2, axis=-1))
            Y_m = (ligand_element_closest != 1) & (ligand_element_closest != 0)
            mask_XY = (D_XY < self.lig_cutoff_for_score) * Y_m[:, 0]  # mask should be all True
        else:
            lig_coords = np.zeros([0, 3], dtype=np.float32)
            lig_elements = np.zeros([0], dtype=np.float32)
            ligand_xyz_closest = np.zeros(
                [Cb_stack.shape[0], self.lig_nn_num, 3], dtype=np.float32
            )
            ligand_element_closest = np.zeros([Cb_stack.shape[0], self.lig_nn_num], dtype=np.int64)
            Y_m = np.full((Cb_stack.shape[0], self.lig_nn_num), False, dtype=bool)
            mask_XY = np.full(Cb_stack.shape[0], False, dtype=bool)

        out_dict = dict(
            X=X_stack,
            X_m=side_chain_mask_stack,
            Y=ligand_xyz_closest,
            Y_t=ligand_element_closest,
            Y_coords=lig_coords,
            S=seq_int_stack,
            seq=[restype_INTtoSTR[i] for i in seq_int_stack],
            R_idx=res_idx_stack,
            chain_labels=chain_labels_stack,
            # for inference
            chain_letters=chain_letters_list,
            ins_codes=results["atom_array"][
                results["atom_array"].atom_name == "CA"
            ].ins_code.tolist(),
            mask=np.ones([X_stack.shape[0]], dtype=bool),
            Y_m=Y_m,
            mask_XY=mask_XY,
        )
        for key in out_dict:
            assert key not in results, f"Key {key} already exists in results"
        results.update(out_dict)
        return results


class BuildInferenceInput(BaseTransform):
    def transform(self, results: Dict) -> Dict | Tuple[List] | None:

        # renumber the residue index
        R_idx_original = copy.deepcopy(results["R_idx"])
        R_idx_list = []
        count = 0
        R_idx_prev = -100000
        for R_idx in list(results["R_idx"]):
            if R_idx_prev == R_idx:
                count += 1
            R_idx_list.append(R_idx + count)
            R_idx_prev = R_idx
        R_idx_renumbered = np.array(R_idx_list, dtype=np.int64)
        results["R_idx"] = R_idx_renumbered
        results["R_idx_original"] = np.array(R_idx_original, dtype=np.int64)
        return results


class RandomSelectAssembly(BaseTransform):

    def transform(self, results: Dict) -> Dict | Tuple[List] | None:
        """
        Required keys: "assembly_len_dict"
        Added keys: "assembly_id"
        """
        assembly_ids = list(results["assembly_len_dict"].keys())
        assembly_id = np.random.choice(assembly_ids)
        results["assembly_id"] = assembly_id
        return results


class ChooseFirstAssembly(BaseTransform):
    """Choose the first assembly for test"""

    def transform(self, results: Dict) -> Dict | Tuple[List] | None:
        results["assembly_id"] = list(results["assembly_len_dict"].keys())[0]
        return results


class PackInputs(BaseTransform):
    def __init__(self, keys: Sequence[str]):
        self.keys = keys

    def transform(self, results):
        out_dict = {}
        for key in self.keys:
            out_dict[key] = results[key]
        return out_dict


class RandomChoosePlddtPair(BaseTransform):
    """
    Random choose a pair of designs with plddt difference larger than min_gap
    Args:
        min_gap (float): The minimum gap between two designs
        min_chosen_plddt (float): The minimum plddt of the chosen design
        max_rejected_plddt (float): The maximum plddt of the rejected design
    """

    def __init__(self, min_gap=1.0, min_chosen_plddt=None, max_rejected_plddt=None):
        """
        Args:
            min_gap (float): The minimum gap between two designs, 0 - 100
        """
        self.min_gap = min_gap
        self.min_chosen_plddt = min_chosen_plddt
        self.max_rejected_plddt = max_rejected_plddt

    def transform(self, results: Dict):
        designs = results["designs"]
        designs = sorted(designs, key=lambda x: x["plddt"])
        n = len(designs)
        all_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if designs[j]["plddt"] - designs[i]["plddt"] > self.min_gap:
                    all_pairs.append((i, j))
        if len(all_pairs) == 0:
            raise ValueError(f"No pair found: {results['cif_path']}")

        if self.min_chosen_plddt is not None:
            all_pairs = [
                (i, j) for i, j in all_pairs if designs[j]["plddt"] >= self.min_chosen_plddt
            ]
        if self.max_rejected_plddt is not None:
            all_pairs = [
                (i, j) for i, j in all_pairs if designs[i]["plddt"] <= self.max_rejected_plddt
            ]

        i, j = random.choice(all_pairs)
        results["chosen"] = designs[j]
        results["rejected"] = designs[i]
        return results


class ChooseMaxMinPlddtPair(BaseTransform):
    """Choose the pair of designs with the largest plddt difference"""

    def transform(self, results: Dict):
        designs = results["designs"]
        designs = sorted(designs, key=lambda x: x["plddt"])
        n = len(designs)
        i, j = 0, n - 1
        results["chosen"] = designs[j]
        results["rejected"] = designs[i]
        return results
