import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import os
import os.path


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(dataset_name, opt, mode):

        # INPUT = IMAGES:
        if dataset_name == 'images_SMPL':
            from data.images_SMPL import Dataset
            dataset = Dataset(opt, mode)
        elif dataset_name == 'overfit_demo_image_SMPL':
            from data.overfit_demo_image_SMPL import Dataset
            dataset = Dataset(opt, mode)
        # INPUT = VOXELS:
        elif dataset_name == 'voxels_SMPL':
            from data.voxels_SMPL import Dataset
            dataset = Dataset(opt, mode)
        # INPUT = VOXELS OF HANDS:
        elif dataset_name == 'voxels_MANO':
            from data.voxels_MANO import Dataset
            dataset = Dataset(opt, mode)

        else:
            raise ValueError("Dataset [%s] not recognized." % dataset_name)

        print('Dataset {} was created'.format(dataset.name))
        return dataset

class DatasetBase(data.Dataset):
    def __init__(self, opt, mode):
        super(DatasetBase, self).__init__()
        self._name = 'BaseDataset'
        self._root = None
        self._opt = opt
        self._mode = mode
        self._create_transform()

        self._IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._root

    def _create_transform(self):
        self._transform = transforms.Compose([])

    def get_transform(self):
        return self._transform

    def _is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self._IMG_EXTENSIONS)

    def _is_csv_file(self, filename):
        return filename.endswith('.csv')

    def _get_all_files_in_subfolders(self, dir, is_file):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

        return images
