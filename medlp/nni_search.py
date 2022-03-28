import os
import click
import json
import yaml
import shutil
import logging
from pathlib import Path
from types import SimpleNamespace as sn
from utils_cw import Print, get_items_from_file, check_dir

from medlp.models import get_engine
from medlp.data_io.dataio import get_dataloader, DATASET_MAPPING
from medlp.utilities.utils import detect_port, parse_nested_data
from medlp.utilities.enum import Phases
from medlp.utilities.click_ex import get_nni_exp_name

from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from ignite.engine import Events
from monai_ex.handlers import CheckpointLoader
from monai_ex.utils import optional_import, min_version

# Need NNI package
nni, has_nni = optional_import("nni", "1.8", min_version)
if has_nni:
    from nni.utils import merge_parameter


@click.command("train-nni")
@click.option("--config", type=click.Path(exists=True), help="Config file to load")
def train_nni(**kwargs):
    logger = logging.getLogger("nni_search")
    logger.info("-" * 10 + "start" + "-" * 10)
    configures = get_items_from_file(kwargs["config"], format="json")
    configures["nni"] = True
    cargs = sn(**configures)

    try:
        # get parameters from tuner
        tuner_params = nni.get_next_parameter()
        logger.info(f"tuner_params: {tuner_params}")

        exp_id = nni.get_experiment_id()
        trial_id = nni.get_trial_id()
        cargs.experiment_path = os.path.join(cargs.experiment_path, exp_id, 'trials', trial_id)

        nested_params = parse_nested_data(tuner_params)
        # cargs = merge_parameter(cargs, tuner_params)
        cargs = merge_parameter(cargs, nested_params)
        logger.info(f"Current args: {cargs}")
        Print('Args:', cargs, color='y')
        
        try:
            cargs.gpu_ids = list(range(len(list(map(int, cargs.gpus.split(","))))))
        except ValueError as e:
            # temp solution for MIG env
            cargs.gpu_ids = list(range(len(list(map(str, cargs.gpus.split(","))))))

        data_list = DATASET_MAPPING[cargs.framework][cargs.tensor_dim][cargs.data_list]["PATH"]
        assert os.path.isfile(data_list), "Data list not exists!"
        files_list = get_items_from_file(data_list, format="auto")
        if cargs.partial < 1:
            logger.info(f"Use {int(len(files_list)*cargs.partial)} data")
            files_list = files_list[: int(len(files_list) * cargs.partial)]
        cargs.split = int(cargs.split) if cargs.split > 1 else cargs.split
        files_train, files_valid = train_test_split(
            files_list, test_size=cargs.split, random_state=cargs.seed
        )
        logger.info(
            f"Get {len(files_train)} training data,"
            f"{len(files_valid)} validation data"
        )

        # Save param and datalist
        with open(os.path.join(cargs.experiment_path, "train_files.yml"), "w") as f:
            yaml.dump(files_train, f)
        with open(os.path.join(cargs.experiment_path, "test_files.yml"), "w") as f:
            yaml.dump(files_valid, f)

        train_loader = get_dataloader(cargs, files_train, phase=Phases.TRAIN)
        valid_loader = get_dataloader(cargs, files_valid, phase=Phases.VALID)

        # Tensorboard Logger
        writer = SummaryWriter(
            log_dir=os.path.join(cargs.experiment_path, "tensorboard")
        )

        trainer, net = get_engine(
            cargs,
            train_loader,
            valid_loader,
            writer=writer
        )

        trainer.add_event_handler(
            event_name=Events.EPOCH_STARTED, handler=lambda x: print("\n", "-" * 40)
        )
        if os.path.isfile(cargs.pretrained_model_path):
            logger.info(
                f"Load pretrained model for contiune training:\n"
                f"\t {cargs.pretrained_model_path}"
            )
            trainer.add_event_handler(
                event_name=Events.STARTED,
                handler=CheckpointLoader(
                    load_path=cargs.pretrained_model_path,
                    load_dict={"net": net},
                    strict=False,
                    skip_mismatch=True,
                ),
            )
        trainer.run()
    except Exception as exception:
        logger.exception(exception)
        raise


@click.command("nni-search")
@click.option("--param-list", type=click.Path(exists=True), help="Base hyper-param setting (.json)")
@click.option("--search-space", type=click.Path(exists=True), help="NNI search space file (.json)")
@click.option("--nni-config", type=str, default="", help="NNI config file (.yaml)")
@click.option("--port", type=int, default=8080, help="Port of nni server")
@click.option("--background", is_flag=True, help="Run nni in background")
@click.option("--out-dir", type=str, prompt=True, show_default=True, default="/homes/clwang/Data/medlp_exp/NNI")
@click.option("--gpus", prompt="Choose GPUs[eg: 0]", type=str, help="The ID of active GPU")
@click.option("--experiment-path", type=str, callback=get_nni_exp_name, default="nni-search")
def nni_search(**args):
    cargs = sn(**args)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        Print(f"CUDA_VISIBLE_DEVICES specified: {os.environ['CUDA_VISIBLE_DEVICES']}, ignoring --gpu flag", color='y')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cargs.gpus)
    gpus_ = os.environ["CUDA_VISIBLE_DEVICES"]

    configures = get_items_from_file(args["param_list"], format="json")
    configures["out_dir"] = cargs.out_dir
    configures["gpus"] = gpus_
    configures["nni"] = True
    configures["experiment_path"] = cargs.experiment_path
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(cargs.gpus)

    if not os.path.isfile(cargs.nni_config):
        nni_config_path = Path(__file__).parent.joinpath("misc/nni_template_config.yml")
        nni_config = get_items_from_file(nni_config_path, format="auto")
    else:
        nni_config = get_items_from_file(cargs.nni_config, format="auto")

    main_file = Path(__file__).parent.joinpath("main.py")
    paramlist_file = os.path.join(cargs.experiment_path, "param.list")
    nniconfig_file = os.path.join(cargs.experiment_path, "nni_config.yml")
    searchspace_file = os.path.join(cargs.experiment_path, "search_space.json")
    nni_config["trial"]["command"] = f"""\
        CUDA_VISIBLE_DEVICES={gpus_} \
        python {str(main_file)} train-nni \
        --config {str(paramlist_file)}"""

    nni_config["searchSpacePath"] = searchspace_file
    nni_config["logDir"] = cargs.experiment_path
    nni_config["experimentName"] = os.path.basename(cargs.experiment_path)
    # nni_config['localConfig']['gpuIndices'] = str(cargs.gpus)

    check_dir(cargs.experiment_path)
    with open(paramlist_file, "w") as f:
        json.dump(configures, f, indent=2)
    with open(nniconfig_file, "w") as f:
        yaml.dump(nni_config, f, default_flow_style=False)
    shutil.copyfile(cargs.search_space, searchspace_file)

    port = cargs.port
    while detect_port(port):
        Print(
            f"Port {port} is used by another process, automatically change to {port+1}!",
            color="y",
        )
        port += 1

    if not cargs.background:
        command = f"nnictl create --config {nniconfig_file} --port {port} --foreground"
    else:
        command = f"nnictl create --config {nniconfig_file} --port {port}"

    try:
        os.system(command)
    except KeyboardInterrupt:
        print("Experimented is terminated by user!")
        os.system(f"nnictl stop --port {port}")
