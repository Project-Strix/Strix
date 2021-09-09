from medlp.utilities.registry import Registry

TRAIN_ENGINES = Registry()
TEST_ENGINES = Registry()
ENSEMBLE_TEST_ENGINES = Registry()

from medlp.models.cnn.engines.segmentation_engines import *
from medlp.models.cnn.engines.classification_engines import *
from medlp.models.cnn.engines.selflearning_engines import *
from medlp.models.cnn.engines.multitask_engines import *
from medlp.models.cnn.engines.siamese_engines import *
