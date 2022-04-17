from monai.networks.nets import UNet, DynUNet
from monai.optimizers import Novograd
from monai.networks import one_hot
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import torch
import os

class UnetModel():

    def __init__(self, args):

        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.device = torch.device('cuda', self.local_rank)
        
        self.model_name = args.model

        if self.model_name == "dyunet":
            kernels, strides = self.dyunet_get_info(args)
            self.model = DynUNet(spatial_dims=3, in_channels=1, out_channels=3, kernel_size=kernels, strides=strides,
                upsample_kernel_size=strides[1:], norm_name="instance", deep_supervision=True, deep_supr_num=args.deep_supr_num)
        else:
            self.model = UNet(spatial_dims=3, in_channels=1, out_channels=3, channels=(16, 32, 64, 128, 256),
                     strides=(2, 2, 2, 2), num_res_units=2, kernel_size=3, up_kernel_size=3)

        self.loss_fn = DiceCELoss(include_background=False, softmax=True, reduction='sum', to_onehot_y=True)
        self.optimizer = Novograd(self.model.parameters(), lr=5e-3)

        self.model.to(self.device)
        self.loss_fn.to(self.device)

        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        self.checkpoints = args.checkpoints
        self.name = args.name
        if args.load:
            self.model.load_state_dict(torch.load(f"{self.checkpoints}/{self.name}.pth", map_location={'cuda:%d' % 0: 'cuda:%d' % self.local_rank}))

        self.scaler = GradScaler()
        self.acm_grad = args.acm_grad
        self.sw_batch_size = int(args.samples*args.batch_size/args.mini_batch)
        self.roi = args.roi

        self.to_device = args.to_device

    @staticmethod
    def dyunet_get_info(args):

        spacings = args.spacing
        sizes = args.roi

        strides, kernels = [], []

        while True:
            spacing_ratio = [sp / min(spacings) for sp in spacings]
            stride = [
                2 if ratio <= 2 and size >= 8 else 1
                for (ratio, size) in zip(spacing_ratio, sizes)
            ]
            kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
            if all(s == 1 for s in stride):
                break
            sizes = [i / j for i, j in zip(sizes, stride)]
            spacings = [i * j for i, j in zip(spacings, stride)]

            kernels.append(kernel)
            strides.append(stride)

        strides.insert(0, len(spacings) * [1])
        kernels.append(len(spacings) * [3])

        return kernels, strides

    def train(self,image,label,step):

        self.model.train()

        with autocast():

            result = self.model(image)

            if self.model_name == "dyunet":
                result = torch.unbind(result, dim=1)
                loss = sum([0.5**i*self.loss_fn(p, label) for i,p in enumerate(result)]) / self.acm_grad
                result = result[0]
            else:
                loss = self.loss_fn(result, label) / self.acm_grad

        self.scaler.scale(loss).backward()

        if step:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            for param in self.model.parameters():
                param.grad = None

        scores = self.dice_score(result,label)
        scores.append(loss.item()*self.acm_grad)

        return scores

    def eval(self,image,label):

        self.model.eval()

        with torch.no_grad():

            if self.model_name == "dyunet" : self.model.deep_supervision = False

            result = sliding_window_inference(inputs=image, roi_size=self.roi, sw_batch_size=self.sw_batch_size, 
                            predictor=self.model, sw_device=self.device, device=self.device)

            if self.model_name == "dyunet" : self.model.deep_supervision = True

            loss = self.loss_fn(result, label) / self.acm_grad

        scores = self.dice_score(result,label)
        scores.append(loss.item()*self.acm_grad)

        return scores

    @staticmethod
    def dice_score(inputs,targets):

        B,N,H,W,D = inputs.shape

        inputs=torch.argmax(inputs,dim=1).view(B,1,H,W,D)
        inputs = one_hot(inputs,num_classes=N)
        targets = one_hot(targets,num_classes=N)

        assert list(inputs.shape) == list(targets.shape)

        scores=[B]
        '''
        i = 0, background
        i = 1, body
        i = 2, tumor
        '''
        for i in range(1,N):

            _target = targets[:,i,:,:,:].reshape(B, -1)
            _input  = inputs[:,i,:,:,:].reshape(B, -1)

            intersection = (_input * _target).sum(1)
            union = _input.sum(1) + _target.sum(1)
            dice = (2. * intersection) / (union + 1e-8)
            scores.append(dice.sum().item())

        return scores

    def save(self):
        torch.save(self.model.state_dict(), f"{self.checkpoints}/{self.name}.pth")