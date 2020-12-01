import argparse

def get_gain_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-type',
        choices=['mnist', 'letter', 'spam'],
        default='mnist',
        type=str)
    parser.add_argument(
        '--learning-rate',
        help='learning rate of model training',
        default=0.01,
        type=float)
    parser.add_argument(
        '--workers',
        default=4,
        type=int,
        help="number of workers for the data loader"
    )
    parser.add_argument(
        '--max-epochs',
        default=1000,
        type=int,
        help="number of workers for the data loader"
    )
    parser.add_argument(
        '--eval-freq',
        default=10,
        type=int,
        help="frequency of evaluations"
    )
    parser.add_argument(
        '--device',
        default="cpu",
        type=str,
        help="device to keep the sensors in",
        choices=["cpu", "cuda"]
    )
    parser.add_argument(
        '--batch-size',
        help='the number of samples in mini batch',
        default=128,
        type=int)
    parser.add_argument(
        '--miss-rate',
        help='mask rate of data',
        default=0.2,
        type=float)
    parser.add_argument(
        '--hint-rate',
        help='rate of hint',
        default=0.9,
        type=float)
    parser.add_argument(
        '--alpha',
        help='weight of mse loss',
        default=100,
        type=float)
    return parser.parse_args()

