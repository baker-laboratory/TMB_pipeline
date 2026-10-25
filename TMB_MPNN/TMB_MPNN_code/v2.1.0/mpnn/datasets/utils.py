def pseudo_single_batch_collate(batch_batch):
    """For already batched data in samplers"""
    assert len(batch_batch) == 1, "Only support batch_size=1"
    return batch_batch[0]
