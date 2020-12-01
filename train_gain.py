from data import get_data_gain
from config import get_gain_args
from experiments import GAINTrainer


def main(args):
    dim, label_dim, train_loader, test_loader, transform_params = get_data_gain(args)
    trainer = GAINTrainer(dim, label_dim, transform_params, args)
    performance_dict = trainer.train_model(train_loader, test_loader)
    return performance_dict


if __name__ == '__main__':
    args = get_gain_args()
    # Call main function
    performance = main(args)
