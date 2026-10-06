"""Download the official datasets used by the ten DTRC tasks."""

import argparse
import os
from pathlib import Path
import shutil
import tarfile

from dtrc.environments.registry import TASKS as OGBENCH


LEWM = {
    "pusht": (
        "quentinll/lewm-pusht",
        "pusht_expert_train.h5.zst",
        "pusht_expert_train.h5",
    ),
    "tworoom": ("quentinll/lewm-tworooms", "tworoom.tar.zst", "tworoom.h5"),
    "reacher": ("quentinll/lewm-reacher", "reacher.tar.zst", "reacher.h5"),
    "cube": (
        "quentinll/lewm-cube",
        "cube_single_expert.tar.zst",
        "ogbench/cube_single_expert.h5",
    ),
}
OGBENCH_URL = "https://rail.eecs.berkeley.edu/datasets/ogbench"


def download_lewm(task, root):
    from huggingface_hub import hf_hub_download
    import zstandard

    repo, filename, relative = LEWM[task]
    destination = root / "datasets" / relative
    if destination.is_file():
        print(f"Already present: {destination}")
        return
    archive = hf_hub_download(
        repo, filename, repo_type="dataset", local_dir=root / "downloads" / task
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    with open(archive, "rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            if filename.endswith(".tar.zst"):
                with tarfile.open(fileobj=reader, mode="r|") as members:
                    found = False
                    for member in members:
                        if (
                            member.isfile()
                            and Path(member.name).name == destination.name
                        ):
                            with (
                                members.extractfile(member) as source,
                                temporary.open("wb") as target,
                            ):
                                shutil.copyfileobj(source, target, length=8 << 20)
                            if temporary.stat().st_size != member.size:
                                raise IOError("Incomplete HDF5 extraction")
                            found = True
                            break
                    if not found:
                        raise ValueError(f"Archive does not contain {destination.name}")
            else:
                with temporary.open("wb") as target:
                    shutil.copyfileobj(reader, target, length=8 << 20)
    os.replace(temporary, destination)
    print(f"Ready: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(LEWM) + tuple(OGBENCH), required=True)
    parser.add_argument("--purpose", choices=("train", "eval", "both"), default="both")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Show official URLs without downloading",
    )
    args = parser.parse_args()
    root = args.data_dir.resolve()
    if args.task in LEWM:
        repo, filename, relative = LEWM[args.task]
        print(f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}")
        print(f"HDF5: {root / 'datasets' / relative}")
        if not args.list_only:
            download_lewm(args.task, root)
        return
    spec = OGBENCH[args.task]
    names = []
    for purpose in ("train", "eval"):
        if args.purpose in (purpose, "both"):
            name = spec[f"{purpose}_state_dataset"].removesuffix(".npz")
            if name not in names:
                names.append(name)
    for name in names:
        print(f"{OGBENCH_URL}/{name}.npz")
        print(f"{OGBENCH_URL}/{name}-val.npz")
    if not args.list_only:
        from ogbench.utils import download_datasets

        # The official downloader fetches the training and validation archives.
        download_datasets(names, dataset_dir=str(root / "ogbench"))
        print(f"Ready: {root / 'ogbench'}")


if __name__ == "__main__":
    main()
