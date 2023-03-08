import argparse
import torch
import torchvision
import numpy as np
from utils.yaml_config_hook import *
from modules import transform, network, models_mae
from evaluation import evaluation
from torch.utils import data

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def inference(loader, model, device):
    model.eval()
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            c = model.forward_cluster(x)
        c = c.detach()
        feature_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        # if step % 20 == 0:
        #     print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    # print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_dataset(args):
    # prepare data
    if args.dataset == "RSOD":
        dataset = torchvision.datasets.ImageFolder(
            root='/home/derek/datasets/RSOD',
            transform=transform.Transforms_BYOL(size=args.image_size),
        )
        class_num = 4
    elif args.dataset == "UC-Merced-Land-Use":
        dataset = torchvision.datasets.ImageFolder(
            root='/home/derek/datasets/UC-Merced-Land-Use',
            transform=transform.Transforms_BYOL(size=args.image_size),
        )
        class_num = 21
    elif args.dataset == "SIRI-WHU":
        dataset = torchvision.datasets.ImageFolder(
            root='/home/derek/datasets/SIRI-WHU',
            transform=transform.Transforms_BYOL(size=args.image_size),
        )
        class_num = 12
    elif args.dataset == "AID":
        dataset = torchvision.datasets.ImageFolder(
            root='/home/derek/datasets/AID',
            transform=transform.Transforms_BYOL(size=args.image_size),
        )
        class_num = 30
    elif args.dataset == "D0-40":
        dataset = torchvision.datasets.ImageFolder(
            root='/home/derek/datasets/D0-40',
            transform=transform.Transforms_BYOL(size=args.image_size),
        )
        class_num = 40
    elif args.dataset == "Chaoyang":
        dataset = torchvision.datasets.ImageFolder(
            root='/home/derek/datasets/Chaoyang/image',
            transform=transform.Transforms_BYOL(size=args.image_size),
        )
        class_num = 4
    else:
        raise NotImplementedError
    return dataset, class_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, class_num = get_dataset(args)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=100,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    mae = models_mae.__dict__["mae_vit_small_patch16"](norm_pix_loss=True)
    model = network.Network_mae(mae, args.feature_dim, class_num)

    for e in range(0, 1001, 50):
        model_fp = os.path.join(args.model_path, "checkpoint_{}.pth".format(e))
        model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
        model.to(device)

        # print("### Creating features from model ###")
        X, Y = inference(data_loader, model, device)
        nmi, ari, f, acc = evaluation.evaluate(Y, X)
        print(f'Epoch[{e}]:', end='')
        print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
