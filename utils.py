import torch
import os
import torch.distributed as dst

def all_gather_reduction(in_data, reduction="sum"):
    if int(os.environ["WORLD_SIZE"]) == 1:
        return in_data

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device('cuda', local_rank)

    data = torch.tensor(in_data, device=device)
    gathers = [torch.zeros_like(data, device=device) for _ in range(world_size)]

    dst.all_gather(tensor_list=gathers, tensor=data)

    sum_gathers = torch.zeros_like(data)
    if reduction == "sum":
        for sample in gathers:
            sum_gathers += sample
        sum_gathers = sum_gathers.tolist()
        return sum_gathers
    elif reduction == "max":
        return max(gathers).item()