from monai.utils.misc import set_determinism
from parser import parse_args
from model import UnetModel
from dataset import niiDataset
from tracer import Tracer
from visualizer import Visualizer
from pathlib import Path
import sys
import torch
import os


class DDP_Engine():

    def __init__(self):

        self.args = parse_args()
        self.setup()
        self.model = UnetModel(self.args)
        self.datasets = niiDataset(self.args)

        self.tracer = Tracer(self.args.name, ["loss", "dice_body", "dice_tumor"])
        self.start_epoch = self.tracer.load() if self.args.load else 0

    def get_tqdm_postfix(self):

        info = {'Body': f"{self.tracer.mode['train'].counter['dice_body'].out() * 100:.2f}%",
                'Tumor': f"{self.tracer.mode['train'].counter['dice_tumor'].out() * 100:.2f}%"}

        return info

    def setup(self):

        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device('cuda', self.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.backends.cudnn.benchmark = True

        if self.local_rank == 0:
            self.vis = Visualizer(self.args)
            Path(self.args.checkpoints).mkdir(parents=True, exist_ok=True)
        else:
            f = open(os.devnull, "w")
            sys.stdout = sys.stderr = f

        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        set_determinism(self.args.seed)

    def all_gather_data(self, in_data):

        if int(os.environ["WORLD_SIZE"]) == 1:
            return in_data

        data = torch.tensor([x for x in in_data]).to(self.device)
        gathers = [torch.zeros_like(data).to(self.device) for _ in range(int(os.environ["WORLD_SIZE"]))]

        torch.distributed.all_gather(tensor_list=gathers, tensor=data)

        sum_gathers = torch.zeros_like(data)
        for sample in gathers:
            sum_gathers += sample

        sum_gathers = sum_gathers.tolist()

        torch.distributed.barrier()

        return sum_gathers

    def train(self):

        for batch_id, batch in enumerate(self.datasets.train_loader, 1):

            N = batch['image'].shape[0]

            for i in range(self.args.mini_batch):

                s = int(i * N / self.args.mini_batch)
                e = int((i + 1) * N / self.args.mini_batch)

                if not self.args.to_device:
                    image, label = (batch['image'][s:e].to(self.device), batch['label'][s:e].to(self.device))
                else:
                    image, label = (batch['image'][s:e], batch['label'][s:e])

                data = self.all_gather_data(self.model.train(image, label,
                                                             batch_id % self.args.acm_grad == 0 or batch_id == len(
                                                                 self.datasets.train_loader)))

                self.tracer.mode['train'].counter['loss'].add(data[-1], data[0])
                self.tracer.mode['train'].counter['dice_body'].add(data[1], data[0])
                self.tracer.mode['train'].counter['dice_tumor'].add(data[2], data[0])

        self.model.scheduler.step()

    def eval(self):

        for batch_id, batch in enumerate(self.datasets.test_loader, 1):

            if not self.args.to_device:
                image, label = (batch['image'].to(self.device), batch['label'].to(self.device))
            else:
                image, label = (batch['image'], batch['label'])

            data = self.all_gather_data(self.model.eval(image, label))

            self.tracer.mode['test'].counter['loss'].add(data[-1], data[0])
            self.tracer.mode['test'].counter['dice_body'].add(data[1], data[0])
            self.tracer.mode['test'].counter['dice_tumor'].add(data[2], data[0])

    def visualize_and_save(self, epoch):

        self.tracer.snap_shots(self.start_epoch + epoch + 1, self.args.test_freq)

        if self.local_rank == 0:
            self.vis.plot(self.tracer)
            self.model.save()
            self.tracer.save()

    def final_plot(self):

        if self.local_rank == 0:
            self.tracer.plot()

        torch.distributed.destroy_process_group()
