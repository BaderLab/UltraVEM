"""
Multi-node distributed training script for nnSSL.

This script supports:
1. Single GPU training
2. Single-node multi-GPU training (using mp.spawn)
3. Multi-node multi-GPU training (using SLURM + srun)

For multi-node training, use the accompanying SLURM script.
"""
from datetime import datetime, timedelta
import os,sys
from pathlib import Path
import shutil
import signal
import socket
from typing import Type, Union, Optional
from typing import get_args
from loguru import logger

import nnssl
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnssl.experiment_planning.experiment_planners.plan import Plan, PREPROCESS_SPACING_STYLES
from nnssl.paths import nnssl_preprocessed
from nnssl.run.load_pretrained_weights import load_pretrained_weights
from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from nnssl.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnssl.utilities.find_class_by_name import recursive_find_python_class
from torch.backends import cudnn


def find_free_network_port() -> int:
    """Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def get_dist_info_from_slurm():
    """
    Get distributed training info from SLURM environment variables.
    
    Returns:
        tuple: (rank, local_rank, world_size, master_addr, master_port)
    """
    # Get the global rank (unique across all nodes)
    rank = int(os.environ.get('SLURM_PROCID', 0))
    
    # Get world size (total number of processes)
    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    
    # Get local rank (rank within the node)
    # SLURM_LOCALID is the local process ID within a node
    local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    
    # Get master address from the first node in the job
    if 'MASTER_ADDR' not in os.environ:
        # Get the first node as master
        node_list = os.environ.get('SLURM_NODELIST', 'localhost')
        # Use scontrol to expand the node list and get the first node
        import subprocess
        try:
            result = subprocess.run(
                ['scontrol', 'show', 'hostnames', node_list],
                capture_output=True, text=True, check=True
            )
            master_addr = result.stdout.strip().split('\n')[0]
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: try to parse the node list manually
            # Handle formats like "node[001-004]" or "node001,node002"
            if '[' in node_list:
                # Extract the first node from format like "node[001-004]"
                prefix = node_list.split('[')[0]
                numbers = node_list.split('[')[1].split(']')[0]
                first_num = numbers.split('-')[0].split(',')[0]
                master_addr = f"{prefix}{first_num}"
            elif ',' in node_list:
                master_addr = node_list.split(',')[0]
            else:
                master_addr = node_list
    else:
        master_addr = os.environ['MASTER_ADDR']
    
    # Get master port
    if 'MASTER_PORT' not in os.environ:
        # Use a default port, but make it somewhat unique based on job ID
        job_id = int(os.environ.get('SLURM_JOB_ID', 0))
        master_port = 29500 + (job_id % 1000)  # Range: 29500-30499
    else:
        master_port = int(os.environ['MASTER_PORT'])
    
    return rank, local_rank, world_size, master_addr, master_port


def is_slurm_multinode():
    """Check if we're running in a SLURM multi-node environment."""
    return 'SLURM_PROCID' in os.environ and 'SLURM_NTASKS' in os.environ


def get_trainer_from_args(
    dataset_name_or_id: Union[int, str],
    configuration: str,
    fold: int,
    trainer_name: str = "nnsslTrainer",
    plans_identifier: str = "nnsslPlans",
    device: torch.device = torch.device("cuda"),
):
    # load nnunet class and do sanity checks
    nnssl_trainer_cls: Type[AbstractBaseTrainer] = recursive_find_python_class(
        join(nnssl.__path__[0], "training", "nnsslTrainer"), trainer_name, "nnssl.training.nnsslTrainer"
    )
    if nnssl_trainer_cls is None:
        raise RuntimeError(
            f"Could not find requested nnunet trainer {trainer_name} in "
            f"nnssl.training.nnsslTrainer ("
            f'{join(nnssl.__path__[0], "training", "nnsslTrainer")}). If it is located somewhere '
            f"else, please move it there."
        )
    assert issubclass(nnssl_trainer_cls, AbstractBaseTrainer), (
        "The requested nnunet trainer class must inherit from " "nnsslTrainer"
    )

    # handle dataset input. If it's an ID we need to convert to int from string
    if dataset_name_or_id.startswith("Dataset"):
        pass
    else:
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError(
                f"dataset_name_or_id must either be an integer or a valid dataset name with the pattern "
                f"DatasetXXX_YYY where XXX are the three(!) task ID digits. Your "
                f"input: {dataset_name_or_id}"
            )

    # initialize nnunet trainer
    preprocessed_dataset_folder_base = join(nnssl_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + ".json")
    plans: Plan = Plan.load_from_file(plans_file)
    pretrain_json = load_json(join(preprocessed_dataset_folder_base, f"pretrain_data__{configuration}.json"))
    nnssl_trainer: AbstractBaseTrainer = nnssl_trainer_cls(
        plan=plans,
        configuration_name=configuration,
        fold=fold,
        pretrain_json=pretrain_json,
        device=device,
    )
    return nnssl_trainer


def maybe_load_checkpoint(
    nnunet_trainer: AbstractBaseTrainer,
    continue_training: bool,
    validation_only: bool,
    pretrained_weights_file: str = None,
):
    if continue_training and pretrained_weights_file is not None:
        raise RuntimeError(
            "Cannot both continue a training AND load pretrained weights. Pretrained weights can only "
            "be used at the beginning of the training."
        )

    if continue_training:
        logger.info("Attempting to continue training...")
        expected_checkpoint_file = join(nnunet_trainer.output_folder, "checkpoint_final.pth")
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, "checkpoint_latest.pth")
        # special case where --c is used to run a previously aborted validation
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, "checkpoint_best.pth")
        if isfile(expected_checkpoint_file):
            logger.info(f"Using {expected_checkpoint_file} as the starting checkpoint for training...")
        else:
            expected_checkpoint_file = None
            logger.info(f"No starting checkpoint available, starting a new training...")
    elif validation_only:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, "checkpoint_final.pth")
        if not isfile(expected_checkpoint_file):
            raise RuntimeError(f"Cannot run validation because the training is not finished yet!")
    else:
        if pretrained_weights_file is not None:
            if not nnunet_trainer.was_initialized:
                nnunet_trainer.initialize()
            load_pretrained_weights(nnunet_trainer.network, pretrained_weights_file, verbose=True)
        expected_checkpoint_file = None

    if expected_checkpoint_file is not None:
        try:
            nnunet_trainer.load_checkpoint(expected_checkpoint_file)
        except EOFError:
            os.remove(expected_checkpoint_file)


def setup_ddp(rank, world_size, master_addr=None, master_port=None):
    """
    Initialize the distributed process group.
    
    For multi-node training, master_addr and master_port should be provided.
    For single-node training with mp.spawn, they should be set in environment variables.
    """
    if master_addr is not None:
        os.environ['MASTER_ADDR'] = master_addr
    if master_port is not None:
        os.environ['MASTER_PORT'] = str(master_port)
    
    # Initialize the process group with NCCL backend
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=30)  # Increased timeout for multi-node
    )


def cleanup_ddp():
    dist.destroy_process_group()


def run_ddp(
    rank,
    dataset_name_or_id,
    configuration,
    fold,
    tr,
    p,
    disable_checkpointing,
    c,
    val,
    pretrained_weights,
    npz,
    val_with_best,
    world_size,
):
    """
    Run DDP training - used for single-node multi-GPU with mp.spawn.
    """
    setup_ddp(rank, world_size)
    torch.cuda.set_device(torch.device("cuda", dist.get_rank()))

    device = torch.device(f"cuda:{rank}")
    nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, tr, p, device)
    if disable_checkpointing:
        nnunet_trainer.disable_checkpointing = disable_checkpointing

    assert not (c and val), f"Cannot set --c and --val flag at the same time. Dummy."

    maybe_load_checkpoint(nnunet_trainer, c, val, pretrained_weights)

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    # Prepare the auto-exiting in case wall-time is exceeded.
    signal.signal(signal.SIGUSR1, nnunet_trainer.exit_training)

    if not val:
        nnunet_trainer.run_training()

    if val_with_best:
        nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, "checkpoint_best.pth"))
    nnunet_trainer.perform_actual_validation(npz)
    cleanup_ddp()


def run_ddp_multinode(
    dataset_name_or_id,
    configuration,
    fold,
    tr,
    p,
    disable_checkpointing,
    c,
    val,
    pretrained_weights,
    npz,
    val_with_best,
):
    """
    Run DDP training for multi-node setup using SLURM.
    This function is called once per process (one per GPU across all nodes).
    """
    # Get distributed info from SLURM
    rank, local_rank, world_size, master_addr, master_port = get_dist_info_from_slurm()
    
    # Log distributed setup info (only on rank 0)
    if rank == 0:
        logger.info(f"Multi-node DDP training initialized:")
        logger.info(f"  World size: {world_size}")
        logger.info(f"  Master address: {master_addr}")
        logger.info(f"  Master port: {master_port}")
        logger.info(f"  Nodes: {os.environ.get('SLURM_NNODES', 'unknown')}")
        logger.info(f"  GPUs per node: {os.environ.get('SLURM_GPUS_PER_NODE', 'unknown')}")
    
    # Initialize DDP
    setup_ddp(rank, world_size, master_addr, master_port)
    
    # Set the local GPU device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        logger.info(f"Rank {rank}: Using device {device}")
    
    # Create trainer
    nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, tr, p, device)
    
    if disable_checkpointing:
        nnunet_trainer.disable_checkpointing = disable_checkpointing

    assert not (c and val), f"Cannot set --c and --val flag at the same time."

    maybe_load_checkpoint(nnunet_trainer, c, val, pretrained_weights)

    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True

    # Prepare the auto-exiting in case wall-time is exceeded.
    signal.signal(signal.SIGUSR1, nnunet_trainer.exit_training)

    if not val:
        nnunet_trainer.run_training()

    if val_with_best:
        nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, "checkpoint_best.pth"))
    
    # Only run validation on rank 0 to avoid duplicate outputs
    if rank == 0:
        nnunet_trainer.perform_actual_validation(npz)
    
    # Synchronize before cleanup
    dist.barrier()
    cleanup_ddp()


def run_training(
    dataset_name_or_id: Union[str, int],
    configuration: str,
    fold: Union[int, str],
    trainer_class_name: str = "nnsslTrainer",
    plans_identifier: str = "nnsslPlans",
    pretrained_weights: Optional[str] = None,
    num_gpus: int = 1,
    export_validation_probabilities: bool = False,
    continue_training: bool = False,
    only_run_validation: bool = False,
    disable_checkpointing: bool = False,
    val_with_best: bool = False,
    device: torch.device = torch.device("cuda"),
    use_slurm_multinode: bool = False,
):
    if isinstance(fold, str):
        if fold != "all":
            try:
                fold = int(fold)
            except ValueError as e:
                print(
                    f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!'
                )
                raise e

    if val_with_best:
        assert not disable_checkpointing, "--val_best is not compatible with --disable_checkpointing"

    # Check if we're in a SLURM multi-node environment
    if use_slurm_multinode or is_slurm_multinode():
        logger.info("Detected SLURM multi-node environment, using SLURM-based DDP")
        run_ddp_multinode(
            dataset_name_or_id,
            configuration,
            fold,
            trainer_class_name,
            plans_identifier,
            disable_checkpointing,
            continue_training,
            only_run_validation,
            pretrained_weights,
            export_validation_probabilities,
            val_with_best,
        )
    elif num_gpus > 1:
        # Single-node multi-GPU using mp.spawn
        assert (
            device.type == "cuda"
        ), f"DDP training (triggered by num_gpus > 1) is only implemented for cuda devices. Your device: {device}"

        os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ.keys():
            port = str(find_free_network_port())
            print(f"using port {port}")
            os.environ["MASTER_PORT"] = port

        mp.spawn(
            run_ddp,
            args=(
                dataset_name_or_id,
                configuration,
                fold,
                trainer_class_name,
                plans_identifier,
                disable_checkpointing,
                continue_training,
                only_run_validation,
                pretrained_weights,
                export_validation_probabilities,
                val_with_best,
                num_gpus,
            ),
            nprocs=num_gpus,
            join=True,
        )
    else:
        # Single GPU training
        nnunet_trainer = get_trainer_from_args(
            dataset_name_or_id,
            configuration,
            fold,
            trainer_class_name,
            plans_identifier,
            device=device,
        )

        signal.signal(signal.SIGUSR1, nnunet_trainer.exit_training)

        if disable_checkpointing:
            nnunet_trainer.disable_checkpointing = disable_checkpointing

        assert not (
            continue_training and only_run_validation
        ), f"Cannot set --c and --val flag at the same time. Dummy."

        maybe_load_checkpoint(nnunet_trainer, continue_training, only_run_validation, pretrained_weights)

        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        if not only_run_validation:
            nnunet_trainer.run_training()

        if val_with_best:
            nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, "checkpoint_best.pth"))
        nnunet_trainer.perform_actual_validation(export_validation_probabilities)


def run_training_entry():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_name_or_id", type=str, help="Dataset name or ID to train with")
    parser.add_argument(
        "configuration",
        type=str,
        help="Configuration that should be trained",
        choices=get_args(PREPROCESS_SPACING_STYLES),
    )
    parser.add_argument(
        "-tr",
        type=str,
        required=False,
        default="nnsslTrainer",
        help="[OPTIONAL] Use this flag to specify a custom trainer. Default: nnsslTrainer",
    )
    parser.add_argument(
        "-p",
        type=str,
        required=False,
        default="nnsslPlans",
        help="[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnSSLPlans",
    )
    parser.add_argument(
        "-pretrained_weights",
        type=str,
        required=False,
        default=None,
        help="[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model. Will only "
        "be used when actually training. Beta. Use with caution.",
    )
    parser.add_argument(
        "-num_gpus", type=int, default=1, required=False, 
        help="Specify the number of GPUs to use for training (single-node only)"
    )
    parser.add_argument(
        "--npz",
        action="store_true",
        required=False,
        help="[OPTIONAL] Save softmax predictions from final validation as npz files (in addition to predicted "
        "segmentations). Needed for finding the best ensemble.",
    )
    parser.add_argument(
        "--c", action="store_true", required=False, help="[OPTIONAL] Continue training from latest checkpoint"
    )
    parser.add_argument(
        "--val",
        action="store_true",
        required=False,
        help="[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.",
    )
    parser.add_argument(
        "--val_best",
        action="store_true",
        required=False,
        help="[OPTIONAL] If set, the validation will be performed with the checkpoint_best instead "
        "of checkpoint_final. NOT COMPATIBLE with --disable_checkpointing! "
        "WARNING: This will use the same 'validation' folder as the regular validation "
        "with no way of distinguishing the two!",
    )
    parser.add_argument(
        "--disable_checkpointing",
        action="store_true",
        required=False,
        help="[OPTIONAL] Set this flag to disable checkpointing. Ideal for testing things out and "
        "you dont want to flood your hard drive with checkpoints.",
    )
    parser.add_argument(
        "-device",
        type=str,
        default="cuda",
        required=False,
        help="Use this to set the device the training should run with. Available options are 'cuda' "
        "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
        "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!",
    )
    parser.add_argument(
        "--slurm_multinode",
        action="store_true",
        required=False,
        help="[OPTIONAL] Force use of SLURM multi-node DDP. Usually auto-detected from environment.",
    )
    args = parser.parse_args()

    assert args.device in [
        "cpu",
        "cuda",
        "mps",
    ], f"-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}."

    # ------------------------------- Post Parsers ------------------------------- #

    dataset_name = args.dataset_name_or_id
    config = args.configuration

    assert (
        os.environ.get("nnssl_results") is not None
    ), "nnssl_results not set. Stopping as no outputs would be written otherwise."

    if args.device == "cpu":
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device("cpu")
    elif args.device == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device("cuda")
    else:
        device = torch.device("mps")
    
    run_training(
        dataset_name,
        config,
        "all",
        args.tr,
        args.p,
        args.pretrained_weights,
        args.num_gpus,
        args.npz,
        args.c,
        args.val,
        args.disable_checkpointing,
        args.val_best,
        device=device,
        use_slurm_multinode=args.slurm_multinode,
    )


if __name__ == "__main__":
    run_training_entry()