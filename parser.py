import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="SphereFace")

    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--device', type=str, default="cpu")

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eval_interval', type=int, default=2)

    parser.add_argument('--train_file', type=str, default="./data/pairsDevTrain.txt")
    parser.add_argument('--eval_file', type=str, default="./data/pairsDevTest.txt")
    parser.add_argument('--img_folder', type=str, default="./data/lfw")

    return parser.parse_args()
