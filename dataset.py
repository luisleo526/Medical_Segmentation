from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
                              CropForegroundd, Spacingd, RandRotated, RandZoomd, RandGaussianSmoothd,
                              RandScaleIntensityd, RandShiftIntensityd,
                              RandGaussianNoised, RandFlipd, RandCropByPosNegLabeld, EnsureTyped, ToDeviced)
from monai.data import CacheDataset, CacheNTransDataset
from monai.data import partition_dataset, ThreadDataLoader
import os
import torch
import shutil
import glob


class niiDataset():

    def __init__(self, args):

        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.device = torch.device('cuda', self.local_rank)

        if args.reset_cache and self.local_rank == 0:
            self.reset_cache()

        self.make_data(args)

        self.train_loader = self.make_loader(args, self.TrData, True)
        self.test_loader = self.make_loader(args, self.TsData, False)

    def reset_cache(self):
        try:
            shutil.rmtree("cache")
        except:
            pass

    def make_data(self, args):

        self.TrData = []
        for image, label in zip(glob.glob(f"{args.train_data}/images/*"), glob.glob(f"{args.train_data}/labels/*")):
            self.TrData.append({'image': image, 'label': label})

        self.TrData = partition_dataset(data=self.TrData, num_partitions=int(os.environ["WORLD_SIZE"]), shuffle=True,
                                        even_divisible=True, drop_last=False)[self.rank]

        self.TsData = []
        for image, label in zip(glob.glob(f"{args.test_data}/images/*"), glob.glob(f"{args.test_data}/labels/*")):
            self.TsData.append({'image': image, 'label': label})

        self.TsData = partition_dataset(data=self.TsData, num_partitions=int(os.environ["WORLD_SIZE"]), shuffle=True,
                                        even_divisible=True, drop_last=False)[self.rank]

    def get_transform(self, args, training=False):

        BASIC = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(keys="image", a_min=-150, a_max=220, b_min=0, b_max=1, clip=True),
            Spacingd(keys=["image", "label"], pixdim=args.spacing, mode=("bilinear", "nearest"),
                     align_corners=[True, True]),
            CropForegroundd(keys=["image", "label"], source_key="image")
        ]

        NOISE = []
        if training:
            NOISE = [
                RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3,
                            mode=['bilinear', 'nearest'], prob=0.2),
                RandZoomd(keys=["image", "label"], min_zoom=0.8, max_zoom=1.2, mode=["trilinear", "nearest"],
                          align_corners=[True, False], prob=0.15),
                RandGaussianSmoothd(keys="image", sigma_x=[0.5, 1.15], sigma_y=[0.5, 1.15], sigma_z=[0.5, 1.15],
                                    prob=0.15),
                RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                RandGaussianNoised(keys="image", std=0.01, prob=0.15),
                RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
                RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
                RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
                RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=args.roi,
                                       num_samples=args.samples, pos=1.0, neg=1.0, image_key="image",
                                       image_threshold=0.0)
            ]

        ENSURE = [EnsureTyped(keys=["image", "label"])]

        transform = BASIC + NOISE + ENSURE

        if args.to_device:
            transform.append(ToDeviced(keys=["image", "label"], device=self.device))

        return Compose(transform)

    def make_loader(self, args, data, training):

        if args.cached:
            dataset = CacheDataset(data=data, transform=self.get_transform(args, training))
        else:
            dataset = CacheNTransDataset(data=data, transform=self.get_transform(args.training))

        loader = ThreadDataLoader(dataset, batch_size=args.batch_size if training else 1, num_workers=0)

        return loader
