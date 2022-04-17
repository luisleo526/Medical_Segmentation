from engine import DDP_Engine
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm import trange
import torch

@record
def main():

    Engine = DDP_Engine()

    epochs = trange(Engine.args.num_epoch, desc="Epoch")
    for epoch in epochs:

        Engine.tracer.reset()
        Engine.train()
        if (Engine.start_epoch+epoch+1) % Engine.args.test_freq == 0 : Engine.eval()
        Engine.visualize_and_save(epoch)

        epochs.set_postfix(Engine.get_tqdm_postfix())

    Engine.final_plot()

if __name__ == "__main__":
    main()
