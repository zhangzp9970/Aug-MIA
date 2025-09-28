import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms.v2 import *
from torchvision.transforms.v2.functional import *
from tqdm import tqdm
from torchplus.utils import Init

if __name__ == "__main__":
    batch_size = 32
    class_num = 500
    step = 10
    root_dir = "./logZZPMAIN.expand.FaceScrub"
    dataset_dir = "/path/to/FaceScrub1/dataset"
    feature_pkl = "/path/to/target/model/feature_extractor.pkl"
    h = 64
    w = 64

    init = Init(
        seed=9970,
        log_root_dir=None,
        sep=False,
        backup_filename=__file__,
        tensorboard=False,
        comment=f"expand FaceScrub feature",
    )
    output_device = torch.device("cpu")  # init.get_device()
    # log_dir =z init.get_log_dir()
    data_workers = 0

    transform = Compose(
        [
            Resize((h, w)),
            ToImage(),
            ToDtype(torch.float, scale=True),
        ]
    )

    mnist_train_ds = PreProcessFolder(root=dataset_dir, transform=transform)

    mnist_train_ds_len = len(mnist_train_ds)

    priv_ds = mnist_train_ds

    priv_train_dl = DataLoader(
        dataset=priv_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
        pin_memory=True,
    )

    class FeatureExtracter(nn.Module):
        def __init__(self):
            super(FeatureExtracter, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
            self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
            self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
            self.conv4 = nn.Conv2d(256, 512, 3, 1, 1)
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(512)
            self.mp1 = nn.MaxPool2d(2, 2)
            self.mp2 = nn.MaxPool2d(2, 2)
            self.mp3 = nn.MaxPool2d(2, 2)
            self.mp4 = nn.MaxPool2d(2, 2)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.relu4 = nn.ReLU()
            self.relu5 = nn.ReLU()

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.mp1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.mp2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.mp3(x)
            x = self.relu3(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.mp4(x)
            x = self.relu4(x)
            x = x.view(-1, 8192)
            return x

    feature_extractor = FeatureExtracter().to(output_device).train(False)

    assert os.path.exists(feature_pkl)
    feature_extractor.load_state_dict(
        torch.load(open(feature_pkl, "rb"), map_location=output_device)
    )

    feature_extractor.requires_grad_(False)

    with torch.no_grad():
        os.makedirs(root_dir, exist_ok=True)
        for i, ((im, _, properties), label) in enumerate(
            tqdm(priv_train_dl, desc=f"expand features")
        ):
            feature8192list = []
            im = im.to(output_device)
            bs, c, h, w = im.shape
            if bs == 1:
                continue
            feature8192 = feature_extractor.forward(im)
            feature8192 = F.normalize(feature8192)
            feature8192list.append(feature8192)
            if step > 1:
                for b in range(bs):
                    for b1 in range(b + 1, bs):
                        feature8192i = torch.zeros((step - 1, feature8192.size(1))).to(
                            output_device
                        )
                        delta = (feature8192[b1] - feature8192[b]) / step
                        for e in range(1, step):
                            feature8192i[e - 1] = feature8192[b] + delta * e
                        feature8192i = F.normalize(feature8192i)
                        feature8192list.append(feature8192i)
            feature8192list = torch.cat(feature8192list)
            with open(os.path.join(root_dir, f"myfeatures_{i}.pkl"), "wb") as f:
                torch.save(feature8192list, f)
