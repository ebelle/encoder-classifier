from collections import OrderedDict
from random import shuffle
from torch.utils.data import Sampler
import linecache


class BucketBatchSampler(Sampler):
    def __init__(self, filepath, batch_size):
        self.batch_size = batch_size
        ind_n_len = []
        total_data = sum(1 for _ in open(filepath, "r")) - 1
        for i in range(total_data):
            text = linecache.getline(filepath, i + 2).split("\t")[1]
            ind_n_len.append((i, len(text)))
        self.ind_n_len = ind_n_len
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    def _generate_batch_map(self):
        # shuffle all of the indices first
        shuffle(self.ind_n_len)
        # Organize lengths
        batch_map = OrderedDict()
        for idx, length in self.ind_n_len:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # split indices into batches of equal size
        batch_list = []
        for length, indices in batch_map.items():
            for group in [
                indices[i : (i + self.batch_size)]
                for i in range(0, len(indices), self.batch_size)
            ]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.ind_n_len)

    def __iter__(self):
        self.batch_list = self._generate_batch_map()
        # shuffle batches
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i
