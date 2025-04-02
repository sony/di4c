import ml_collections
import numpy as np

def get_config(subset=False):

    if subset:
        subset_size = 100

    cifar10_path = 'cifar10/datasets'

    config = ml_collections.ConfigDict()
    config.eval_name = 'CIFAR_elbo'
    config.save_location = 'cifar10/elbo'
    config.train_config_overrides = [
        [['device'], 'cuda'],
        [['data', 'root'], cifar10_path],
        [['distributed'], False]
    ]
    config.experiment_dir = 'cifar10'
    config.checkpoint_path = 'cifar10/checkpoints/ckpt_0001999999.pt'


    config.loggers = ['ELBO']

    config.device = 'cuda'

    config.data = data = ml_collections.ConfigDict()
    data.name = 'DiscreteCIFAR10'
    data.root = cifar10_path
    data.train = False
    data.download = True
    data.S = 256
    data.batch_size = 16
    data.shuffle = False
    data.shape = [3,32,32]
    data.random_flips = False
    data.subset = subset
    if subset:
        np.random.seed(242065)
        data.indices = np.random.choice(10000, size=subset_size)


    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = 'TauLeaping'
    sampler.num_steps = 1000
    sampler.min_t = 0.02
    sampler.eps_ratio = 1e-9
    sampler.finish_strat = 'max'
    sampler.theta = 1.0
    sampler.initial_dist = 'gaussian'
    sampler.num_corrector_steps = 1
    sampler.corrector_step_size_multiplier = 1.0
    sampler.corrector_entry_time = 1.0

    config.logging = logging = ml_collections.ConfigDict()
    logging.total_N = 100
    logging.total_B = 10000
    if subset:
        logging.total_B = subset_size

    logging.B = 50
    logging.min_t = 0.01
    logging.eps = 1e-9
    logging.initial_dist = 'gaussian'

    return config