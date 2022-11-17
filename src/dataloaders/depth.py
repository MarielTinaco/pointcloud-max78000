import sys, os
import zipfile
import shutil
from pathlib import Path

import torch
import torchvision
from torchvision import transforms

import ai8x

DEPTHMAP_FILENAME = "DepthMap"
DEPTHMAP_ZIP_SOURCE = Path(__file__).parent.parent.parent / 'data' / f'{DEPTHMAP_FILENAME}.zip'

def depthmap_get_datasets (data, load_train=True, load_test=True):

    (data_dir, args) = data
    path = data_dir
    dataset_path = os.path.join(path, DEPTHMAP_FILENAME)
    is_dir = os.path.isdir(dataset_path)

    train_path = Path(dataset_path) / 'train'
    test_path = Path(dataset_path) / 'test'

    if not is_dir:

        with zipfile.ZipFile(DEPTHMAP_ZIP_SOURCE, 'r') as zip_ref:
            zip_ref.extractall(Path(dataset_path).parent)

        dataset_path = Path(dataset_path)
        
        Path.mkdir (train_path)
        Path.mkdir (test_path)

        for dir in os.listdir(dataset_path):
            
            if dir in set(['test', 'train']):
                continue

            for subdir in os.listdir(dataset_path / dir):

                sourcedir = dataset_path / dir / subdir
                destdir = dataset_path / subdir / dir

                if not Path.exists(destdir):
                    Path.mkdir(destdir)

                for files in os.listdir(sourcedir):
                    
                    source = sourcedir / files
                    destination = destdir / files

                    shutil.move(source, destination)

        # Clean up
        for dir in os.listdir(dataset_path):
            if dir not in set(['test', 'train']):
                shutil.rmtree(dataset_path / dir)

        print(f"DepthMap dataset has been loaded and moved to {str(dataset_path)}")

    # Loading and normalizing train dataset
    if load_train:
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = torchvision.datasets.ImageFolder(root=str(train_path),
                                                        transform=train_transform)
    else:
        train_dataset = None

    # Loading and normalizing test dataset
    if load_test:
        test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = torchvision.datasets.ImageFolder(root=str(test_path),
                                                        transform=test_transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'DepthMap',
        'input': (1, 128, 128),
        'output': ('bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet'),
        'loader': depthmap_get_datasets,
    },
]
