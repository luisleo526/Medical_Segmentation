from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=15)

    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--reset_cache", action='store_true')
    parser.add_argument("--checkpoints", type=str, default="checkpoints")

    parser.add_argument("--roi", type=int, nargs='+')
    parser.add_argument("--spacing", type=float, nargs='+')
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--act", type=str, default="softmax")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mini_batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--acm_grad", type=int, default=1)
    parser.add_argument("--num_epoch", type=int, default=1000)

    parser.add_argument("--cached", action='store_true')
    parser.add_argument("--to_device", action='store_true')
    parser.add_argument("--model", type=str, default="unet", help='options: "unet", "dyunet" "unetr')
    parser.add_argument("--features_size", type=int, default=16)
    parser.add_argument("--deep_supr_num", type=int, default=3)
    parser.add_argument("--test_freq", type=int, default=25)

    parser.add_argument("--load", action='store_true')
    parser.add_argument("--name", type=str, default='PAN')

    parser.add_argument("--display_port", type=int, default=8097)

    args = parser.parse_args()

    return args
