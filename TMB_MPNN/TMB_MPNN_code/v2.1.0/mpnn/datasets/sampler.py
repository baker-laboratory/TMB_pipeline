
import numpy as np
import pandas as pd
from torch.utils.data import BatchSampler


def make_batch_iter(
    df,
    pdb_lengths_dict,
    cluster_data_cls,
    batch_tokens,
    length_cutoff,
    max_number_of_pdbs=100000,
    date_cutoff="2030-Jan-01",
):
    """
    inputs:
    df - panda data frame, e.g. pd.read_csv()
    pdb_lengths - dictionary {"PDB_id": {"biounit_id": assembly_length}}; e.g. np.load("valid_2.5A_ls.npz", allow_pickle=True)['arr_0'].item()
    cluster_data_cls - StructureLoader class;
    batch_tokens [int] - number of tokens for one batch;
    outputs:
    """
    id_list = list(df["label"])
    degree = list(df["degree"])
    date_list = list(df["date"])
    protein_chain_letters = list(df["protein"])
    rand_perm = list(np.random.permutation(len(id_list)))
    sampled_labels = []
    for i in rand_perm:
        label = id_list[i]
        prob = 1.0 / (1.0 + degree[i])
        rand_uniform = np.random.uniform(0.0, 1.0, size=1)
        date_flag = date_list[i] < date_cutoff
        if (len(protein_chain_letters[i]) > 2) and (prob > rand_uniform) and date_flag:
            sampled_labels.append(label)

    L_list = []
    name_list = []
    for sample in sampled_labels:
        if sample in pdb_lengths_dict:
            temp_dict = pdb_lengths_dict[sample]  # biounits with lengths
            biounit_list = list(temp_dict.keys())
            k = len(biounit_list)
            if k > 1:
                idx = np.random.randint(0, high=k, size=1, dtype=int)[0]
            else:
                idx = 0
            chosen_biounit_key = biounit_list[idx]
            current_length = temp_dict[chosen_biounit_key]
            if current_length >= length_cutoff \
                and len(L_list) < max_number_of_pdbs:
                L_list.append(current_length)
                name_list.append((sample, chosen_biounit_key, current_length))

    structure_loader = cluster_data_cls(name_list, L_list, max_tokens_per_batch=batch_tokens)

    batch_iter = []
    for _, batch in enumerate(structure_loader):
        batch_iter.append(batch)
    iter_num = len(batch_iter)
    batch_iter = iter(batch_iter)
    return batch_iter, iter_num


class StructureLoader:
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
            new_batch_size = max(batch_max, size) * (len(batch) + 1)
            if new_batch_size <= self.max_tokens_per_batch:
                batch.append(ix)
                batch_max = max(batch_max, size)
            else:
                if len(batch) > 0:
                    clusters.append(batch)
                # start a new batch
                if size <= self.max_tokens_per_batch:
                    batch, batch_max = [ix], size
                else:
                    batch, batch_max = [], 0

            # if size * (len(batch) + 1) <= self.max_tokens_per_batch:
            #     batch.append(ix)
            #     batch_max = size
            # else:
            #     if len(batch) > 0:
            #         clusters.append(batch)
            #     batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)

        # verify that the total number of tokens in each batch is less than max_tokens_per_batch
        for b_ids in clusters:
            total_tokens = sum([self.lengths[i] for i in b_ids])
            assert total_tokens <= self.max_tokens_per_batch

        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_ids in self.clusters:
            batch = [self.dataset[i] for i in b_ids]
            yield batch

class MPNNSampler(BatchSampler):
    """Adopt from ligand_run_v.py"""

    def __init__(
        self,
        df_path,
        npz_path,
        batch_tokens,
        min_protein_len_cutoff,
        max_num_of_pdb_valid,
        date_cutoff,
        **kwargs,
    ):
        print('MPNNSampler get unused kwargs:', kwargs)
        self.df = pd.read_csv(df_path)
        self.npz_data = np.load(npz_path, allow_pickle=True)["arr_0"].item()
        self.sample_num = len(self.df)

        self.batch_tokens = batch_tokens
        self.min_protein_len_cutoff = min_protein_len_cutoff
        self.max_num_of_pdb_valid = max_num_of_pdb_valid
        self.date_cutoff = date_cutoff

        batch_iter, iter_num = self.reset()
        super().__init__(batch_iter, batch_size=1, drop_last=False)

    def reset(self):
        batch_iter, iter_num = make_batch_iter(
            self.df, self.npz_data, StructureLoader, self.batch_tokens,
            self.min_protein_len_cutoff, self.max_num_of_pdb_valid, self.date_cutoff
        )
        self.sampler = batch_iter
        self.iter_num = iter_num
        return batch_iter, iter_num

    def __len__(self) -> int:
        """Replace len(self.sampler) with self.iter_num."""
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return self.iter_num // self.batch_size  # type: ignore[arg-type]
        else:
            return (self.iter_num + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]

    def __iter__(self):
        return super().__iter__()
