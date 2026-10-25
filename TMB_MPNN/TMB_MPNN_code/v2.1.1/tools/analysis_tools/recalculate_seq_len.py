
from tqdm import tqdm
import torch
import os
import multiprocessing
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def process_entry(args):
    pdb_id, assembly_id, seq_dict = args
    subfolder = pdb_id[1:3]
    cache_folder = f'/net/scratch/faxue/mpnn_preprocessed_TrRosetta/{subfolder}'
    cache_path = f'{cache_folder}/{pdb_id}_{assembly_id}.pth'
    if not os.path.exists(cache_path):   # use original data
        seq_len = seq_dict[assembly_id]
    else:
        preprocessed_data = torch.load(cache_path)
        seq_len = len(preprocessed_data['S'])
    seq_gap = seq_len - seq_dict[assembly_id]
    # if abs(seq_gap) > 20:
    #     print(f'{pdb_id}_{assembly_id}: {seq_dict[assembly_id]} -> {seq_len}, diff: {seq_gap}')
    return pdb_id, assembly_id, seq_len, seq_gap


if __name__ == '__main__':
    subset = 'valid_3.5A_ls'
    npz_path = f'data/{subset}.npz'
    data_list = np.load(npz_path, allow_pickle=True)['arr_0'].item()
    new_data_list = {}
    seq_gap_list = []
    total_tasks = sum([len(seq_dict) for seq_dict in data_list.values()])

    arg_list = []
    for pdb_id, seq_dict in data_list.items():
        for assembly_id, seq_len in seq_dict.items():
            arg_list.append((pdb_id, assembly_id, seq_dict))

    with multiprocessing.Pool(multiprocessing.cpu_count() * 2) as pool:
        results = list(tqdm(pool.imap(process_entry, arg_list), total=total_tasks))

    for pdb_id, assembly_id, seq_len, seq_gap in results:
        if pdb_id not in new_data_list:
            new_data_list[pdb_id] = {}
        new_data_list[pdb_id][assembly_id] = seq_len
        seq_gap_list.append(seq_gap)

    # save seq_gap to csv file
    with open('output/seq_gap.csv', 'w') as f:
        f.write('pdb_id,assembly_id,new_seq_len,seq_len_diff\n')
        for pdb_id, assembly_id, seq_len, seq_gap in results:
            f.write(f'{pdb_id},{assembly_id},{seq_len},{seq_gap}\n')

    # plot dist of seq gap
    plt.figure()
    plt.hist(seq_gap_list, log=True, bins=100)
    # sns.histplot(seq_gap_list, binwidth=10, 
    #              log_scale=(False, True),
    #              )
    plt.xlabel('Sequence length difference')
    plt.ylabel('Frequency')
    plt.title('Sequence length difference distribution')
    plt.savefig('output/seq_gap.png')

    np.savez_compressed(f'data/{subset}_reCalLen.npz', new_data_list)
