from medlp.utilities.utils import ENGINES, assert_network_type
from medlp.models.rcnn.modeling.detector.generalized_rcnn import GeneralizedRCNN

@ENGINES.register('detection')
def build_detection_engine(**kwargs):
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

    assert_network_type(opts.model_type, 'RCNN')

    _DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)