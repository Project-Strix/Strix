from strix.models.cnn.engines import TRAIN_ENGINES, TEST_ENGINES, ENSEMBLE_TEST_ENGINES
from monai_ex.engines import GanTrainer

@TRAIN_ENGINES.register('gan')
def build_gan_engine(**kwargs):
    opts = kwargs['opts']
    train_loader = kwargs['train_loader']
    test_loader = kwargs['test_loader']
    net = kwargs['net']
    loss = kwargs['loss']
    optim = kwargs['optim']
    lr_scheduler = kwargs['lr_scheduler']
    writer = kwargs['writer']
    valid_interval = kwargs['valid_interval']
    device = kwargs['device']
    model_dir = kwargs['model_dir']
    logger_name = kwargs.get('logger_name', None)

    GanTrainer(
        device=device,
        max_epochs=opts.n_epoch,
        train_data_loader=train_loader,
        g_network=,
        g_optimizer=,
        g_loss_function=,
        d_network=,
        d_optimizer=,
        d_loss_function=,
        epoch_length=int(opts.n_epoch_len),
    )