import gc
import os
import sys
import tempfile
from typing import Any

import torch
from fvcore.nn import FlopCountAnalysis
from prettytable import PrettyTable
from torch.profiler import ProfilerActivity, profile

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_DIR, "src"))
torch.set_float32_matmul_precision("medium")

from glasses_detector import GlassesClassifier, GlassesDetector, GlassesSegmenter


def check_filesize(model: torch.nn.Module):
    with tempfile.NamedTemporaryFile() as temp:
        torch.save(model.state_dict(), temp.name)
        return os.path.getsize(temp.name) / (1024**2)


def check_num_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_flops(model: torch.nn.Module, input: Any):
    flops = FlopCountAnalysis(model, (input,))
    return flops.total()


def check_ram(model: torch.nn.Module, input: Any):
    # Clean up
    gc.collect()

    with profile(
        activities=[ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        # Run model
        model(input)

    # Sum the memory usage of all the operations on the CPU
    memory_usage = sum(item.cpu_memory_usage for item in prof.key_averages())

    return memory_usage / (1024**2)


def check_vram(model: torch.nn.Module, input: Any):
    # Clean up
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Check the allocated memory before
    mem_before = torch.cuda.max_memory_allocated()

    if isinstance(input, list):
        # Detection models require a list
        input = [torch.tensor(i, device="cuda") for i in input]
    else:
        # Otherwise a single input is enough
        input = torch.tensor(input, device="cuda")

    # Run the model on CUDA
    model = model.to("cuda")
    model(input)
    torch.cuda.synchronize()

    # Check the allocated memory after
    mem_after = torch.cuda.max_memory_allocated()

    # Clean up
    model.to("cpu")
    torch.cuda.empty_cache()

    return (mem_after - mem_before) / (1024**2)


def analyse(model_cls, task):
    # Create a table
    table = PrettyTable()
    field_names = [
        "Model size",
        "Filesize (MB)",
        "Num params",
        "FLOPS",
        "RAM (MB)",
    ]

    if torch.cuda.is_available():
        # If CUDA is available, add VRAM
        field_names.append("VRAM (MB)")

    # Set the field names
    table.field_names = field_names

    if task == "detection":
        # Detection models require a list
        input = [*torch.randn(1, 3, 256, 256)]
    else:
        # Otherwise a single input is enough
        input = torch.randn(1, 3, 256, 256)

    for size in ["small", "medium", "large"]:
        # Clean up
        gc.collect()

        # Load the model without pre-trained weights on the CPU
        model = model_cls(size=size, weights=False, device="cpu").model
        model.eval()

        # Check the basic stats
        filesize = check_filesize(model)
        num_params = check_num_params(model)
        flops = check_flops(model, input)
        ram = check_ram(model, input)

        # Stats
        row = [
            size,
            f"{filesize:.2f}",
            f"{num_params:,}",
            f"{flops:,}",
            f"{ram:.2f}",
        ]

        if torch.cuda.is_available():
            # If CUDA is available, check VRAM
            vram = check_vram(model, input)
            row.append(f"{vram:.2f}")

        # Add the stats row
        table.add_row(row)

    # Print table
    print(table)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        required=True,
        choices=["classification", "detection", "segmentation"],
    )
    return parser.parse_args()


def main():
    args = parse_args()

    match args.task:
        case "classification":
            model_cls = GlassesClassifier
        case "detection":
            model_cls = GlassesDetector
        case "segmentation":
            model_cls = GlassesSegmenter

    analyse(model_cls, args.task)


if __name__ == "__main__":
    main()
