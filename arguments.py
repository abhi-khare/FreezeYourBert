from argparse import ArgumentParser


def arguments():
    parser = ArgumentParser()

    # model architecture parameter
    parser.add_argument("--model", type=str, default="roberta-base")
    parser.add_argument("--dropout", type=float, default="0.2")
    parser.add_argument("--layer_dim", type=int, default="256")

    # dataset params
    parser.add_argument("--tokenizer", type=str, default="roberta-base")
    parser.add_argument("--dataset", type=str, required=True)

    # training params
    parser.add_argument("--precision", type=int, default="16")
    parser.add_argument("--seed", type=int, default="42")
    parser.add_argument("--deterministic", type=bool, default=True)
    parser.add_argument("--max_epoch", type=int, default=6)

    args = parser.parse_args()
    return args
