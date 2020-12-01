from .mnist import get_mnist, get_mnist_gain
from .uci import get_uci

__datagetters_gain__ = {"mnist": get_mnist_gain,
                        "spam": get_uci,
                        "letter": get_uci}

def get_data_gain(args):
    if args.data_type in __datagetters_gain__:
        train_loader, test_loader, transform_params = __datagetters_gain__[args.data_type](args)
        dim = train_loader.dataset.input_size
        label_dim = train_loader.dataset.output_size
        return dim, label_dim, train_loader, test_loader, transform_params
    else:
        raise NotImplementedError(args.data_type)

