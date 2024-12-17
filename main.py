import time

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from model import SphereCNN
from dataloader import LFW4Training, LFW4Eval
from parser import parse_args
from utils import set_seed, AverageMeter


def eval(data_loader: DataLoader, model: SphereCNN, device: torch.device, threshold: float = 0.5):
    model.eval()
    model.feature = True
    sim_func = nn.CosineSimilarity()

    cnt = 0.
    total = 0.

    t1 = time.time()
    with torch.no_grad():
        for img_1, img_2, label in data_loader:
            img_1 = img_1.to(device)
            img_2 = img_2.to(device)
            label = label.to(device)

            feat_1 = model(img_1, None)
            feat_2 = model(img_2, None)
            sim = sim_func(feat_1, feat_2)

            sim[sim > threshold] = 1
            sim[sim <= threshold] = 0

            total += sim.size(0)
            for i in range(sim.size(0)):
                if sim[i] == label[i]:
                    cnt += 1

    print("Acc.: %.4f; Time: %.3f" % (cnt / total, time.time() - t1))
    return


def main():
    args = parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    train_set = LFW4Training(args.train_file, args.img_folder)
    eval_set = LFW4Eval(args.eval_file, args.img_folder)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size)

    model = SphereCNN(class_num=train_set.n_label)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_record = AverageMeter()
    for epoch in range(args.epoch):
        t1 = time.time()
        model.train()
        model.feature = False
        loss_record.reset()

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            _, loss = model(inputs, targets)
            loss.backward()
            optimizer.step()

            loss_record.update(loss)

        print("Epoch: %s; Loss: %.3f; Time: %.3f" % (str(epoch).zfill(2), loss_record.avg, time.time() - t1))

        if (epoch + 1) % args.eval_interval == 0:
            eval(eval_loader, model, device)

    return


if __name__ == "__main__":
    main()
