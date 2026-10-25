import numpy as np
import torch
import itertools
import time

class PDBDataset(torch.utils.data.Dataset):
    def __init__(self,
                 cif_path,
                 parser,
                 atom_list_to_save=['N', 'CA', 'C', 'O'],
                 parse_only_DNA_RNA=0,
                 protein_backbone_occ_cutoff=0.8,
                 protein_side_chain_occ_cutoff=0.5,
                 excluded_elements=[1],
                 ligand_occ_cutoff=0.01,
                 num_nn = 25,
                 parse_only_proteins=0):

        self.num_nn = num_nn
        self.ligand_occ_cutoff = ligand_occ_cutoff
        self.excluded_elements = excluded_elements
        self.protein_backbone_occ_cutoff = protein_backbone_occ_cutoff
        self.protein_side_chain_occ_cutoff = protein_side_chain_occ_cutoff
        self.parse_only_DNA_RNA = parse_only_DNA_RNA
        self.parse_only_proteins = parse_only_proteins
        if self.parse_only_DNA_RNA:
            self.exclude_v_list = ["nonpoly"]
        else:
            self.exclude_v_list = []
        self.atom_list_to_save = atom_list_to_save
        self.parser = parser
        self.cif_path = cif_path

        self.restype_1to3 = {
            'A': 'ALA',
            'R': 'ARG',
            'N': 'ASN',
            'D': 'ASP',
            'C': 'CYS',
            'Q': 'GLN',
            'E': 'GLU',
            'G': 'GLY',
            'H': 'HIS',
            'I': 'ILE',
            'L': 'LEU',
            'K': 'LYS',
            'M': 'MET',
            'F': 'PHE',
            'P': 'PRO',
            'S': 'SER',
            'T': 'THR',
            'W': 'TRP',
            'Y': 'TYR',
            'V': 'VAL',
        }

        self.restype_3to1 = {v: k for k, v in self.restype_1to3.items()}

        self.alphabet = list('ACDEFGHIKLMNPQRSTVWYX')
        self.alphabet_dict = dict(zip(self.alphabet, range(len(self.alphabet))))

        self.bb_idx_list = []
        self.backbone_list = ["N", "CA", "C", "O"]
        for i, atom in enumerate(self.atom_list_to_save):
            if atom in self.backbone_list:
                self.bb_idx_list.append(i)

    def __getitem__(self, index):
        """
        index = [[("pdb_id", "biounit_id"), ("pdb_id", "biounit_id")]]
        """
        x = [self.loader(pdb_tuple[0], pdb_tuple[1]) for pdb_tuple in index[0]]
        return x

    def loader(self, pdb_id, assembly_id):
        t0 = time.time()
        try:
            chains, asmb, covalei, meta = self.parser.parse(self.cif_path+f'/{pdb_id[1:3]}/{pdb_id}.cif.gz')
        except:
            print('bad_id: ', pdb_id)
            return ("pass", "pass")
        #------------------proteins vs not--------------------
        polypeptide_letter_list = []
        ligand_letter_list = []
        for chain_letter, v in chains.items():
            if v.type == "polypeptide(L)":
                polypeptide_letter_list.append(chain_letter)
            else:
                if v.type not in self.exclude_v_list:
                    ligand_letter_list.append(chain_letter)

        if self.parse_only_proteins:
            ligand_letter_list = []

        atom_list_to_save_dict = dict(zip(self.atom_list_to_save, range(len(self.atom_list_to_save))))
        poly_chain_dict = {}
        extra_ligand_chain_dict = {}
        for poly_chain in polypeptide_letter_list:
            extra_ligand_xyz_list = []
            extra_ligand_element_list = []

            poly_chain_dict[poly_chain] = {}

            L = len(list(set([a[1] for a in list(chains[poly_chain].atoms.keys())])))
            xyz = np.zeros([L, len(self.atom_list_to_save), 3], dtype=np.float32)
            occ = np.zeros([L, len(self.atom_list_to_save)], dtype=np.float32)
            residue_idx = -100*np.ones([L], dtype=np.int32)
            raw_sequence = L*["UNK"]
            for c, (res_id, atom_tuple) in enumerate(itertools.groupby(list(chains[poly_chain].atoms.keys()), lambda x: x[1])):
                for atom_key in atom_tuple:
                    if atom_key[2] not in list(self.restype_3to1):
                        v=chains[poly_chain].atoms[atom_key]
                        if v.element not in self.excluded_elements:
                            if v.occ > self.ligand_occ_cutoff:
                                extra_ligand_xyz_list.append(np.array(v.xyz))
                                extra_ligand_element_list.append(np.array(v.element))

                    else:
                        if atom_key[3] in self.atom_list_to_save:
                            atom_idx = atom_list_to_save_dict[atom_key[3]]
                            xyz[c,atom_idx,:] = np.array(chains[poly_chain].atoms[atom_key].xyz)
                            occ[c,atom_idx] = np.array(chains[poly_chain].atoms[atom_key].occ)
                            if atom_key[3] == 'CA':
                                raw_sequence[c] = atom_key[2]
                                residue_idx[c] = int(atom_key[1])


            if len(extra_ligand_xyz_list) > 0 :
                extra_ligand_chain_dict[poly_chain] = {}
                extra_ligand_chain_dict[poly_chain]["xyz"] = np.stack(extra_ligand_xyz_list)
                extra_ligand_chain_dict[poly_chain]["element"] = np.stack(extra_ligand_element_list)

            mask_occ = occ > self.protein_backbone_occ_cutoff
            mask_occ = (mask_occ[:,self.bb_idx_list[0]]*mask_occ[:,self.bb_idx_list[1]]*mask_occ[:,self.bb_idx_list[2]]*mask_occ[:,self.bb_idx_list[3]]).astype(bool)
            poly_chain_dict[poly_chain]["xyz"] = xyz[mask_occ]
            poly_chain_dict[poly_chain]["occ"] = occ[mask_occ]
            poly_chain_dict[poly_chain]["seq"] = [AA for i, AA in enumerate(raw_sequence) if mask_occ[i]]
            poly_chain_dict[poly_chain]["residue_idx"] = residue_idx[mask_occ]


        ligand_chain_dict = {}
        for letter in ligand_letter_list:
            xyz_list = []
            element_list = []
            for k, v in chains[letter].atoms.items():
                if v.element not in self.excluded_elements:
                    if v.occ > self.ligand_occ_cutoff:
                        xyz_list.append(np.array(v.xyz))
                        element_list.append(np.array(v.element))
            if len(xyz_list) > 0 :
                ligand_chain_dict[letter] = {}
                ligand_chain_dict[letter]["xyz"] = np.stack(xyz_list)
                ligand_chain_dict[letter]["element"] = np.stack(element_list)


        protein_L = 0
        Cb_list = []
        Y_list = []
        Y_t_list = []
        X_list = []
        X_m_list = []
        S_list = []
        R_idx_list = []
        chain_labels_list = []
        chain_multi = 0

        ligand_chain_list = list(ligand_chain_dict.keys())
        poly_chain_list = list(poly_chain_dict.keys())
        extra_ligand_chain_list = list(extra_ligand_chain_dict.keys())

        if assembly_id not in list(asmb.keys()):
            return ("pass", "pass")
        for idx_i, item in enumerate(asmb[assembly_id]):
            if item[0] in poly_chain_list:
                tmp_residue_idx = poly_chain_dict[item[0]]["residue_idx"]
                xyz = poly_chain_dict[item[0]]["xyz"]
                rotation_matrix = item[1][:3,:3]
                translation = item[1][:3,3]
                xyz = np.einsum('ij,raj->rai', rotation_matrix, xyz) + translation[None,None,:]
                protein_L += xyz.shape[0]
                N = xyz[:,self.bb_idx_list[0],:]
                Ca = xyz[:,self.bb_idx_list[1],:]
                C = xyz[:,self.bb_idx_list[2],:]

                b = Ca - N
                c = C - Ca
                a = np.cross(b, c, axis=-1)
                Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
                Cb_list.append(Cb)

                X_list.append(xyz)
                X_m = (poly_chain_dict[item[0]]["occ"] > self.protein_side_chain_occ_cutoff).astype(np.int32)
                X_m_list.append(X_m)

                seq_one_letter = [self.restype_3to1[AA] if AA in list(self.restype_3to1) else "X" for AA in poly_chain_dict[item[0]]["seq"]]
                seq_int = [self.alphabet_dict[AA] if AA in self.alphabet else self.alphabet_dict["X"] for AA in seq_one_letter]

                S_list.append(np.array(seq_int, dtype=np.int32))
                R_idx_list.append(tmp_residue_idx)
                chain_labels_list.append(chain_multi*np.ones_like(tmp_residue_idx, dtype=np.int32))
                chain_multi += 1

                if item[0] in extra_ligand_chain_list:
                    xyz = extra_ligand_chain_dict[item[0]]["xyz"]
                    rotation_matrix = item[1][:3,:3]
                    translation = item[1][:3,3]
                    xyz = np.einsum('ij,rj->ri', rotation_matrix, xyz) + translation[None,:]
                    Y_list.append(xyz)
                    Y_t_list.append(extra_ligand_chain_dict[item[0]]["element"])


            elif item[0] in ligand_chain_list:
                xyz = ligand_chain_dict[item[0]]["xyz"]
                rotation_matrix = item[1][:3,:3]
                translation = item[1][:3,3]
                xyz = np.einsum('ij,rj->ri', rotation_matrix, xyz) + translation[None,:]
                Y_list.append(xyz)
                Y_t_list.append(ligand_chain_dict[item[0]]["element"])


        #Pick closest neighbors per residue using virtual CB distance to context atoms
        Cb_stack = np.concatenate(Cb_list, axis=0) #[L, 3]
        if len(Y_list) != 0:
            Y_stack = np.concatenate(Y_list, axis=0) #[M, 3]
            Y_t_stack = np.concatenate(Y_t_list, axis=0) #[M]

            D = np.sum((Cb_stack[:,None,:]-Y_stack[None,:,:])**2, -1) #[L, M]
            closest_idx = np.argsort(D, -1)[:,:self.num_nn]
            ligand_xyz_closest = np.take_along_axis(np.repeat(Y_stack[None,:,:], Cb_stack.shape[0], 0), np.repeat(closest_idx[:,:,None], 3, -1), axis=1)
            ligand_element_closest = np.take_along_axis(np.repeat(Y_t_stack[None,:], Cb_stack.shape[0], 0), closest_idx, axis=1)
        else:
            ligand_xyz_closest = np.zeros([Cb_stack.shape[0], 1, 3], dtype=np.float32)
            ligand_element_closest = np.zeros([Cb_stack.shape[0], 1], dtype=np.float32)


        X_stack = np.concatenate(X_list, axis=0) #[L, num_atoms, 3]
        X_m_stack = np.concatenate(X_m_list, axis=0) #[L, num_atoms]
        S_stack = np.concatenate(S_list, axis=0) #[L]
        R_idx_stack = np.concatenate(R_idx_list, axis=0) #[L]
        chain_labels_stack = np.concatenate(chain_labels_list, axis=0) #[L]


        out_dict = {}
        out_dict["Y"] = ligand_xyz_closest
        out_dict["Y_t"] = ligand_element_closest

        out_dict["X"] = X_stack
        out_dict["X_m"] = X_m_stack
        out_dict["S"] = S_stack
        out_dict["R_idx"] = R_idx_stack
        out_dict["chain_labels"] = chain_labels_stack
        out_dict["protein_L"] = protein_L

        return (out_dict, protein_L)


class StructureLoader():
    def __init__(self, dataset, protein_lengths, max_tokens_per_batch=2000):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = protein_lengths
        self.max_tokens_per_batch = max_tokens_per_batch
        sorted_ix = np.argsort(self.lengths)
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.max_tokens_per_batch:
                batch.append(ix)
                batch_max = size
            else:
                if len(batch) > 0:
                    clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch


def make_batch_iter(df, pdb_lengths_dict, cluster_data_cls, batch_tokens, length_cutoff, max_number_of_pdbs=100000, date_cutoff="2030-Jan-01"):
    """
    inputs:
    df - panda data frame, e.g. pd.read_csv(f"/home/justas/projects/lab_github/chemnet/arch.22-10-28/pdb/{input_path_name}")
    pdb_lengths - dictionary {"PDB_id": {"biounit_id": assembly_length}}; e.g. np.load("valid_2.5A_ls.npz", allow_pickle=True)['arr_0'].item()
    cluster_data_cls - StructureLoader class;
    batch_tokens [int] - number of tokens for one batch;
    outputs:
    """
    id_list = list(df['label'])
    degree = list(df['degree'])
    date_list = list(df['date'])
    protein_chain_letters = list(df['protein'])
    rand_perm = list(np.random.permutation(len(id_list)))
    sampled_labels=[]
    for i in rand_perm:
        label = id_list[i]
        prob = 1.0/(1.0+degree[i])
        rand_uniform = np.random.uniform(0.0, 1.0, size=1)
        date_flag = date_list[i] < date_cutoff
        if (len(protein_chain_letters[i]) > 2) and (prob>rand_uniform) and date_flag: 
            sampled_labels.append(label)

    pdb_dict_list = list(pdb_lengths_dict)
    L_list = []
    name_list = []
    for sample in sampled_labels:
        if sample in pdb_dict_list:
            temp_dict = pdb_lengths_dict[sample] #biounits with lengths
            biounit_list = list(temp_dict.keys())
            k = len(biounit_list)
            if k > 1:
                idx = np.random.randint(0, high=k, size=1, dtype=int)[0]
            else:
                idx = 0
            chosen_biounit_key = biounit_list[idx]
            current_length = temp_dict[chosen_biounit_key]
            if current_length >= length_cutoff and len(L_list) < max_number_of_pdbs:
                L_list.append(current_length)
                name_list.append((sample,chosen_biounit_key, current_length))

    structure_loader = cluster_data_cls(name_list, L_list, max_tokens_per_batch=batch_tokens)

    batch_iter = []
    for _, batch in enumerate(structure_loader):
        batch_iter.append(batch)
    batch_iter = iter(batch_iter)
    return batch_iter
