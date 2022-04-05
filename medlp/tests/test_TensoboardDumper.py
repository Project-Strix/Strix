import os
import shutil
import tempfile
import unittest

from ignite.engine import Engine, Events
from torch.utils.tensorboard import SummaryWriter

from monai.handlers import TensorBoardStatsHandler
from medlp.models.cnn import *
from monai_ex.handlers import TensorboardDumper


class TestTensorboardDumper(unittest.TestCase):
    def test_image_saver(self):
        pass

    def test_summary_plot(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # set up engine
            def _train_func(engine, batch):
                return batch + 1.0

            engine = Engine(_train_func)

            # set up dummy metric
            @engine.on(Events.EPOCH_COMPLETED)
            def _update_metric(engine):
                current_metric = engine.state.metrics.get("acc", 0.1)
                engine.state.metrics["acc"] = current_metric + 0.1

            # set up testing handler
            writer = SummaryWriter(log_dir=tempdir)
            stats_handler = TensorBoardStatsHandler(
                writer,
                output_transform=lambda x: {"loss": x},
                global_epoch_transform=lambda x: x,
            )
            stats_handler.attach(engine)

            dump_handler = TensorboardDumper(log_dir=tempdir, epoch_level=True)
            dump_handler.attach(engine)

            engine.run(range(10), max_epochs=10)
            writer.close()
            # check logging output
            print("Saved files:", os.listdir(tempdir))
            shutil.copyfile(
                os.path.join(tempdir, "summary.png"),
                "/homes/clwang/test_tb_summary.png",
            )
            self.assertTrue(len(os.listdir(tempdir)) == 2)


if __name__ == "__main__":
    unittest.main()