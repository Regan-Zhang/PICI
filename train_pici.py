import datetime
from pathlib import Path
import time
import torchvision
import argparse
from cluster_pici import inference
from evaluation import evaluation
from modules import transform, network, contrastive_loss, models_mae
from utils.yaml_config_hook import *
from utils.save_model import *
from torch.utils import data
from modules.cross_level_interaction import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class MySampler(object):
    def __init__(self, source, indices):
        self.data_source = source
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.data_source)


def pretrain_one_epoch(pretrain_loader):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(pretrain_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')

        _, loss_1, loss_2 = model(x_i, x_j)
        loss = (loss_1 + loss_2) / 2
        loss.backward()
        optimizer.step()

        # if step % 50 == 0:
        #     print(
        #         f"Step [{step}/{len(data_loader)}]\t loss_restruct1: {loss_1.item()}\t loss_restruct2: {loss_2.item()}")
        loss_epoch += loss.item()
    return loss_epoch


def train_one_epoch(train_loader):
    loss_epoch = 0
    loss_i = 0
    loss_c = 0
    loss_re = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')

        (z_i, z_j, c_i, c_j), loss_1, loss_2 = model(x_i, x_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss = loss_instance + loss_cluster + (loss_1 + loss_2) / 2
        loss.backward()
        optimizer.step()

        # if step % 50 == 0:
        #     print(
        #         f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}\t loss_restruct1: {loss_1.item()}\t loss_restruct2: {loss_2.item()}")
        loss_epoch += loss.item()
        loss_i += loss_instance.item()
        loss_c += loss_cluster.item()
        loss_re += (loss_1.item() + loss_2.item()) / 2

    return loss_epoch, loss_i, loss_c, loss_re


def finetune_one_epoch(loader, new_pseudo_label):
    loss_epoch = 0
    loss_i = 0
    loss_c = 0
    loss_ce = 0
    start_idx, end_idx = 0, args.batch_size
    cross_entropy = torch.nn.CrossEntropyLoss()

    for step, ((x_i, x_j), _) in enumerate(loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')

        (z_i, z_j, c_i, c_j), _, _ = model(x_i, x_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        p = new_pseudo_label[0].numpy().T[0][start_idx:end_idx]
        with torch.no_grad():
            q = c_i.detach().cpu()
            q = torch.argmax(q, dim=1).numpy()
            p_hat = match(p, q)
        loss_ce1 = cross_entropy(c_i, p_hat)
        p = new_pseudo_label[1].numpy().T[0][start_idx:end_idx]
        with torch.no_grad():
            q = c_j.detach().cpu()
            q = torch.argmax(q, dim=1).numpy()
            p_hat = match(p, q)
        loss_ce2 = cross_entropy(c_j, p_hat)
        loss = loss_instance + loss_cluster + (loss_ce1 + loss_ce2) / 2

        loss.backward()
        optimizer.step()
        # if step % 50 == 0:
        #     print(
        #         f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}\t loss_ce1: {loss_ce1.item()}\t loss_ce2: {loss_ce2.item()}")

        loss_epoch += loss.item()
        loss_i += loss_instance.item()
        loss_c += loss_cluster.item()
        loss_ce += (loss_ce1.item() + loss_ce2.item()) / 2

        start_idx += args.batch_size
        end_idx += args.batch_size
    return loss_epoch, loss_i, loss_c, loss_ce


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


def get_val_dataset(args):
    if args.dataset == "RSOD":
        val_dataset = torchvision.datasets.ImageFolder(
            root='/home/derek/datasets/RSOD',
            transform=transform.Transforms_BYOL(size=args.image_size).test_transform,
        )
    elif args.dataset == "UC-Merced-Land-Use":
        val_dataset = torchvision.datasets.ImageFolder(
            root='/home/derek/datasets/UC-Merced-Land-Use',
            transform=transform.Transforms_BYOL(size=args.image_size).test_transform,
        )
    elif args.dataset == "SIRI-WHU":
        val_dataset = torchvision.datasets.ImageFolder(
            root='/home/derek/datasets/SIRI-WHU',
            transform=transform.Transforms_BYOL(size=args.image_size).test_transform,
        )
    elif args.dataset == "AID":
        val_dataset = torchvision.datasets.ImageFolder(
            root='/home/derek/datasets/AID',
            transform=transform.Transforms_BYOL(size=args.image_size).test_transform,
        )
    elif args.dataset == "D0-40":
        val_dataset = torchvision.datasets.ImageFolder(
            root='/home/public/datasets/D0-40',
            transform=transform.Transforms_BYOL(size=args.image_size).test_transform,
        )
    elif args.dataset == "Chaoyang":
        val_dataset = torchvision.datasets.ImageFolder(
            root='/home/derek/datasets/Chaoyang/image',
            transform=transform.Transforms_BYOL(size=args.image_size).test_transform,
        )

    else:
        raise NotImplementedError
    return val_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    # wandb setting
    if args.wandb:
        run_dir = Path("./wandb") / args.project_name
        if not run_dir.exists():
            os.makedirs(str(run_dir))
        import wandb
        wandb.init(dir=run_dir, settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
                   config=args,
                   project=args.project_name,
                   entity=args.entity,
                   name=args.name,
                   )

    dataset, class_num = get_dataset(args)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    val_dataset = get_val_dataset(args)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=args.workers,
    )
    # prepare for cross-level interaction
    indices = torch.randperm(len(dataset))
    mysampler = MySampler(dataset, indices)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        sampler=mysampler,
        num_workers=args.workers,
    )

    mae = models_mae.__dict__["mae_vit_small_patch16"](norm_pix_loss=True)
    model = network.Network_mae(mae, args.feature_dim, class_num)
    # print(model)
    model = model.to('cuda')

    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.pth".format(args.start_epoch))
        print(f"Reloading the model from {model_fp}")
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
    loss_device = torch.device("cuda")
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(
        loss_device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)


    """
    Pre-train for 200 epochs
    Train for 800 epochs
    Boosting for 50 epochs
    """
    start_time = last_time = time.time()
    pretrain_epochs = 200
    max_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        if epoch < pretrain_epochs:
            loss_epoch = pretrain_one_epoch(data_loader)
            types = 'Pretrain'
            if epoch % 100 == 0:
                X, Y = inference(val_loader, model, loss_device)
                nmi, ari, f, acc = evaluation.evaluate(Y, X)
                print(f'Epoch[{epoch}]:', end='')
                print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
            # wandb.log({'Petrain Loss': loss_epoch / len(data_loader)}, step=epoch)
            # wandb.log({'Reconstruction Loss': loss_epoch / len(data_loader)}, step=epoch)
            # wandb.log({'Petrain NMI': nmi}, step=epoch)
            # wandb.log({'Petrain ARI': ari}, step=epoch)
            # wandb.log({'Petrain ACC': acc}, step=epoch)
        elif pretrain_epochs <= epoch < 1000:
            loss_epoch, loss_i, loss_c, loss_re = train_one_epoch(data_loader)
            types = 'Train'
            if epoch % 100 == 0:
                X, Y = inference(val_loader, model, loss_device)
                nmi, ari, f, acc = evaluation.evaluate(Y, X)
                print(f'Epoch[{epoch}]:', end='')
                print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
            # wandb.log({'Train Loss': loss_epoch / len(data_loader)}, step=epoch)
            # wandb.log({'Instance Loss': loss_i / len(data_loader)}, step=epoch)
            # wandb.log({'Cluster Loss': loss_c / len(data_loader)}, step=epoch)
            # wandb.log({'Reconstruction Loss': loss_re / len(data_loader)}, step=epoch)
            # wandb.log({'Train NMI': nmi}, step=epoch)
            # wandb.log({'Train ARI': ari}, step=epoch)
            # wandb.log({'Train ACC': acc}, step=epoch)
        else:
            if epoch == 1000:
                new_pseudo_label = make_pseudo_label(loader, model, class_num)
            loss_epoch, loss_i, loss_c, loss_ce = finetune_one_epoch(loader, new_pseudo_label)
            types = 'Finetune'
            X, Y = inference(val_loader, model, loss_device)
            nmi, ari, f, acc = evaluation.evaluate(Y, X)
            print(f'Epoch[{epoch}]:', end='')
            print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
            # wandb.log({'Finetune Loss': loss_epoch / len(data_loader)}, step=epoch)
            # wandb.log({'Instance Loss': loss_i / len(data_loader)}, step=epoch)
            # wandb.log({'Cluster Loss': loss_c / len(data_loader)}, step=epoch)
            # wandb.log({'Cross-entropy Loss': loss_ce / len(data_loader)}, step=epoch)
            # wandb.log({'Finetune NMI': nmi}, step=epoch)
            # wandb.log({'Finetune ARI': ari}, step=epoch)
            # wandb.log({'Finetune ACC': acc}, step=epoch)
            if acc > max_acc:
                max_acc = acc
                save_model(args, model, optimizer, epoch)
        # wandb.log({'Loss': loss_epoch / len(data_loader)}, step=epoch)
        # wandb.log({'NMI': nmi}, step=epoch)
        # wandb.log({'ARI': ari}, step=epoch)
        # wandb.log({'ACC': acc}, step=epoch)
        # print(f"Epoch [{epoch}/{args.epochs}]\tType:[{types}]\tLoss: {loss_epoch / len(data_loader)}")
        if epoch % 100 == 0 or epoch == 999:
            save_model(args, model, optimizer, epoch)
        epoch_time = time.time() - last_time
        last_time = time.time()
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        # print(f'Epoch[{epoch}] Training time: {epoch_time_str}')

    # wandb.finish()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Total Training time: {total_time_str}')
    save_model(args, model, optimizer, args.epochs)
