from torch.utils.data import ConcatDataset, Dataset
from functools import partial

# all datasets loaded here
from .data_collator import get_data_collator
from .dvc import DenseVideoCaptioningStreamDataset
from .magqa import MAGQAStreamDataset
from .grounding import GroundingStreamDataset

__all__ = [
    'build_concat_train_dataset',
    'build_eval_dataset_dict',
    'get_data_collator',
    'get_compute_metrics_dict'
]

def build_concat_train_dataset_from_config(tokenizer, config):
    datasets = list()
    for dataset_config in config:
        dataset_cls = dataset_config.pop('dataset_cls')
        datasets.append(globals()[dataset_cls](tokenizer=tokenizer, **dataset_config))
    return ConcatDataset(datasets)
