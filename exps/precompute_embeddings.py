import argparse
import os
from typing import Iterable

EMBEDDING_DATASETS = [
    "cifar10-embedding",
    "mnist-embedding",
    "dbpedia-embedding",
    "ag-news-embedding",
    "amazon-review-polarity-embedding",
    "imdb-embedding",
]


def _parse_dataset_items(items: Iterable[str]):
    """
    Supports:
      - name
      - name:train_size:test_size
    """
    out = []
    for raw in items:
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) == 1:
            out.append((parts[0], None, None))
        elif len(parts) == 3:
            name, ts, tes = parts
            out.append((name, int(ts), int(tes)))
        else:
            raise ValueError(
                f"Invalid dataset spec: {raw!r} (expected name or name:train:test)"
            )
    return out


def precompute_embeddings(
    dataset_path: str,
    datasets: Iterable[str],
    train_size: int | None = None,
    test_size: int | None = None,
    random_state: int = 42,
    device: str | None = None,
    image_batch_size: int = 64,
    bert_model_name: str = "bert-base-uncased",
    imdb_max_length: int = 256,
    imdb_batch_size: int = 32,
) -> None:
    """
    Precompute embeddings and write them to disk caches.

    Notes:
    - CIFAR10/MNIST: caches full train/test embeddings once, then future runs only sample from cache.
    - BERT datasets: caches the sampled (train/val/test) embeddings keyed by parameters.
    """
    dataset_path = os.path.abspath(dataset_path)
    if not os.path.isdir(dataset_path):
        raise ValueError(
            f"dataset_path does not exist or is not a directory: {dataset_path}"
        )

    # Import from the same directory (exps/) when running as a script.
    from load_dataset import (
        load_ag_news_dataset,
        load_amazon_review_polarity_dataset,
        load_cifar10_dataset,
        load_dbpedia_ontology_dataset,
        load_imdb_dataset,
        load_mnist_dataset,
    )

    for name, ts_override, tes_override in _parse_dataset_items(datasets):
        ts = train_size if ts_override is None else ts_override
        tes = test_size if tes_override is None else tes_override
        if ts is None or tes is None:
            raise ValueError(
                f"train_size/test_size must be provided (either via --train_size/--test_size or in '{name}:train:test')"
            )

        print(
            f"[precompute] {name} (train_size={ts}, test_size={tes}, seed={random_state})"
        )
        if name == "cifar10-embedding":
            load_cifar10_dataset(
                train_size=ts,
                test_size=tes,
                random_state=random_state,
                batch_size=image_batch_size,
                device=device,
            )
        elif name == "mnist-embedding":
            load_mnist_dataset(
                dataset_path=dataset_path,
                train_size=ts,
                test_size=tes,
                random_state=random_state,
                batch_size=image_batch_size,
                device=device,
            )
        elif name == "dbpedia-embedding":
            load_dbpedia_ontology_dataset(
                dataset_path=dataset_path,
                train_size=ts,
                test_size=tes,
                random_state=random_state,
                model_name=bert_model_name,
                device=device,
            )
        elif name == "ag-news-embedding":
            load_ag_news_dataset(
                dataset_path=dataset_path,
                train_size=ts,
                test_size=tes,
                random_state=random_state,
                model_name=bert_model_name,
                device=device,
            )
        elif name == "amazon-review-polarity-embedding":
            load_amazon_review_polarity_dataset(
                dataset_path=dataset_path,
                train_size=ts,
                test_size=tes,
                random_state=random_state,
                model_name=bert_model_name,
                device=device,
            )
        elif name == "imdb-embedding":
            load_imdb_dataset(
                dataset_path=dataset_path,
                train_size=ts,
                test_size=tes,
                random_state=random_state,
                model_name=bert_model_name,
                device=device,
                max_length=imdb_max_length,
                batch_size=imdb_batch_size,
            )
        else:
            raise ValueError(f"Unknown embedding dataset: {name}")

    print("[precompute] done")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Precompute and cache embedding datasets (run this once before parallel experiments)."
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to data_files directory (e.g. /path/to/project/data_files)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Datasets to precompute: name or name:train:test (e.g. cifar10-embedding:40000:9000)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Precompute all known embedding datasets",
    )
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default=None, help="cuda/cpu (default: auto)"
    )
    parser.add_argument(
        "--image_batch_size",
        type=int,
        default=64,
        help="Batch size for ResNet50 embedding extraction",
    )
    parser.add_argument(
        "--bert_model_name",
        type=str,
        default="bert-base-uncased",
        help="HF model name for BERT embedding datasets",
    )
    parser.add_argument("--imdb_max_length", type=int, default=256)
    parser.add_argument("--imdb_batch_size", type=int, default=32)
    args = parser.parse_args()

    datasets = list(args.datasets)
    if args.all:
        datasets = EMBEDDING_DATASETS
    if not datasets:
        raise SystemExit("Need --datasets ... or --all")

    precompute_embeddings(
        dataset_path=args.dataset_path,
        datasets=datasets,
        train_size=args.train_size,
        test_size=args.test_size,
        random_state=args.seed,
        device=args.device,
        image_batch_size=args.image_batch_size,
        bert_model_name=args.bert_model_name,
        imdb_max_length=args.imdb_max_length,
        imdb_batch_size=args.imdb_batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())