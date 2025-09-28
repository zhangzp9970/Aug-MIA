import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torchvision.datasets import *
from torchvision.transforms.transforms import *
from torchvision.transforms.functional import *
from tqdm import tqdm
from torchplus.utils import (
    Init,
    save_image2,
)


if __name__ == "__main__":
    batch_size = 512
    train_epoches = 100
    log_epoch = 4
    class_num = 530
    root_dir = "./logZZPMAIN.fattack.expand.face"
    feature_pkl = "/path/to/target/model/feature_extractor.pkl"
    data_dir = "/path/to/logZZPMAIN.expand.FaceScrub"
    h = 64
    w = 64
    lr = 0.01
    momentum = 0.9
    weight_decay = 0.0005

    init = Init(
        seed=9970,
        log_root_dir=root_dir,
        sep=True,
        backup_filename=__file__,
        tensorboard=True,
        comment=f"FaceScrub feature attack cos norm newexpand1",
    )
    output_device = init.get_device()
    writer = init.get_writer()
    log_dir, model_dir = init.get_log_dir()
    data_workers = 2

    featureslist = []
    for f in range(1700):
        try:
            features = torch.load(f"{data_dir}/myfeatures_{f}.pkl")
            features = TensorDataset(features)
            featureslist.append(features)
        except:
            break

    priv_ds = ConcatDataset(featureslist)

    # del features
    # del featureslist

    priv_train_dl = DataLoader(
        dataset=priv_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_workers,
        drop_last=True,
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

    class Inversion(nn.Module):
        def __init__(self, in_channels):
            super(Inversion, self).__init__()
            self.in_channels = in_channels
            self.deconv1 = nn.ConvTranspose2d(self.in_channels, 512, 4, 1)
            self.deconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
            self.deconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
            self.deconv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
            self.deconv5 = nn.ConvTranspose2d(64, 3, 4, 2, 1)
            self.bn1 = nn.BatchNorm2d(512)
            self.bn2 = nn.BatchNorm2d(256)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(64)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.relu4 = nn.ReLU()
            self.sigmod = nn.Sigmoid()

        def forward(self, x):
            x = x.view(-1, self.in_channels, 1, 1)
            x = self.deconv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.deconv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.deconv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            x = self.deconv4(x)
            x = self.bn4(x)
            x = self.relu4(x)
            x = self.deconv5(x)
            x = self.sigmod(x)
            return x

    feature_extractor = FeatureExtracter().to(output_device).train(False)
    myinversion = Inversion(8192).train(True).to(output_device)

    assert os.path.exists(feature_pkl)
    feature_extractor.load_state_dict(
        torch.load(open(feature_pkl, "rb"), map_location=output_device)
    )

    feature_extractor.requires_grad_(False)

    optimizer = optim.Adam(
        myinversion.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True
    )

    for epoch_id in tqdm(range(1, train_epoches + 1), desc="Total Epoch"):
        for i, (feature8192) in enumerate(
            tqdm(priv_train_dl, desc=f"epoch {epoch_id}")
        ):
            feature8192 = feature8192[0].to(output_device)
            optimizer.zero_grad()
            rim = myinversion.forward(feature8192)
            rfeature8192 = feature_extractor.forward(rim)
            rfeature8192 = F.normalize(rfeature8192)
            mse = F.mse_loss(feature8192, rfeature8192)
            cos = torch.mean(F.cosine_similarity(feature8192, rfeature8192))
            loss = 1 - cos
            loss.backward()
            optimizer.step()

        if epoch_id % log_epoch == 0:
            writer.add_scalar("mse", mse, epoch_id)
            writer.add_scalar("loss", loss, epoch_id)
            save_image2(rim.detach(), f"{log_dir}/output/{epoch_id}.png")
            with open(
                os.path.join(model_dir, f"myinversion_{epoch_id}.pkl"), "wb"
            ) as f:
                torch.save(myinversion.state_dict(), f)
    writer.close()
