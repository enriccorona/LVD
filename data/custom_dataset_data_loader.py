import torch.utils.data
from data.dataset import DatasetFactory


class CustomDatasetDataLoader:
    def __init__(self, opt, mode=True):
        self._opt = opt
        self._mode = mode
        try:
            self._num_threds = opt.n_threads_train
        except:
            self._num_threds = opt.n_threads_test
        self._create_dataset()

    def _create_dataset(self):
        self._dataset = DatasetFactory.get_by_name(self._opt.dataset_mode, self._opt, self._mode)

        if hasattr(self._dataset, 'collate_fn'):
            self._dataloader = torch.utils.data.DataLoader(
                self._dataset,
                batch_size=self._opt.batch_size,
                collate_fn = self._dataset.collate_fn,
                shuffle=True,
                #shuffle=not self._opt.serial_batches and self._mode == 'train',
                num_workers=int(self._num_threds),
                drop_last=True)
        else:
            self._dataloader = torch.utils.data.DataLoader(
                self._dataset,
                batch_size=self._opt.batch_size,
                shuffle=self._mode != 'test',
                #shuffle=self._mode == 'train',
                #shuffle=True,
                #shuffle=not self._opt.serial_batches and self._mode == 'train',
                num_workers=int(self._num_threds),
                drop_last=True)



    def load_data(self):
        return self._dataloader

    def __len__(self):
        return len(self._dataset)
