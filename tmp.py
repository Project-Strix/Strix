import torch
import monai, random
import numpy as np
from monai.transforms import Randomizable, Compose
from monai.data import Dataset #DataLoader
from torch.utils.data import DataLoader


def wif(worker_id: int) -> None:
    """
    Callback function for PyTorch DataLoader `worker_init_fn`.
    It can set different random seed for the transforms in different workers.

    """
    worker_info = torch.utils.data.get_worker_info()
    if hasattr(worker_info.dataset, "transform") and hasattr(worker_info.dataset.transform, "set_random_state"):
        worker_info.dataset.transform.set_random_state(worker_info.seed % (2 ** 32))


class RandPrint(Randomizable):        
    def randomize(self):
        rnd = self.R.random()
        self.val = rnd
    
    def __call__(self, x):
        self.randomize()
        print(x, self.val)
        return self.val

ds = Dataset(np.arange(12), transform=RandPrint())

loader = DataLoader(ds, batch_size=3, shuffle=False, 
                    num_workers=3, pin_memory=False, worker_init_fn=wif)

for e in range(3):
    print(e, '-'*10)
    #np.random.seed() #not work
    for x in loader:
        pass