from models.SGM.NCSNv2 import *
from utils.util import parse_args_NCSNv2
import torch
from data.Dataloaders import *


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args_NCSNv2()

    size = None

    if args.train:
        train_loader, input_size, channels = pick_dataset(
            dataset_name=args.dataset, 
            batch_size=args.batch_size, 
            normalize=False, 
            size=size, 
            num_workers=args.num_workers
        )

        model = NCSNv2(input_size, channels, args)
        model.train_model(train_loader, args)

    elif args.sample:
        _, input_size, channels = pick_dataset(
            dataset_name=args.dataset, 
            batch_size=args.batch_size, 
            normalize=False, 
            size=size
        )
        model = NCSNv2(input_size, channels, args)
        model.load_checkpoints(args.checkpoint)
        model.sample(args, False)

    else:
        raise ValueError("Invalid mode, choose either train, sample or outlier_detection.")
