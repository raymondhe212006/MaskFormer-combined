"""Top level training script, powered by the lightning CLI."""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pathlib

import comet_ml  # noqa: F401
import torch._inductor.config
torch._inductor.config.shape_padding = False  # Prevents OOM during compile-time pad_mm benchmarking

from lightning.pytorch.cli import ArgsType

from models.maskformer_model import HGCMaskformer
from dataset.dataset_maskformers import HGCDataModule
from hepattn.utils.cli import CLI

#config_dir = pathlib.Path(__file__).parent / "configs"
config_dir = str(pathlib.Path(__file__).resolve().parents[3] / "configs")  # was hard-coded to mmarcheg's tree

def main(args: ArgsType = None) -> None:
    #torch.cuda.set_per_process_memory_fraction(0.6)
    CLI(
        model_class=HGCMaskformer,
        datamodule_class=HGCDataModule,
        args=args,
        parser_kwargs={"default_env": True, "fit": {"default_config_files": [f"{config_dir}/base.yaml"]}},
    )


if __name__ == "__main__":
    main()
