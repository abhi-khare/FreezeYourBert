from argparse import ArgumentParser


def arguments():
    parser = ArgumentParser()
    parser.add_argument("--layer_1_dim", type=int, default=128)
    args = parser.parse_args()
    return args
