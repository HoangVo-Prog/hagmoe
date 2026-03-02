import torch
import os
import numpy as np
import random
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from src.core.data.datasets import AspectSentimentDataset, AspectSentimentDatasetKFold


def get_model(cfg):
    """
    Chuẩn chỉnh:
    - cfg.mode chọn class trong src.models
    - Không fallback (không cfg.base/cfg.moe/cfg.kfold)
    - kwargs chỉ lấy theo signature __init__
    - Báo lỗi rõ nếu thiếu required args
    """
    from src.core.run.model_factory import build_model

    return build_model(cfg)

def get_kfold_dataset(cfg, tokenizer):
    return AspectSentimentDatasetKFold(
            json_path=cfg.train_path,
            tokenizer=tokenizer,
            max_len_sent=cfg.max_len_sent,
            max_len_term=cfg.max_len_term,
            k_folds=cfg.k_folds,
            seed=cfg.seed,
            shuffle=cfg.shuffle,
            mode=getattr(cfg, "mode", None),
            debug_aspect_span=getattr(cfg, "debug_aspect_span", False),
            
        )

def get_dataset(cfg, tokenizer):
    train_set = AspectSentimentDataset(
        json_path=cfg.train_path,
        tokenizer=tokenizer,
        max_len_sent=cfg.max_len_sent,
        max_len_term=cfg.max_len_term,
        mode=getattr(cfg, "mode", None),
        debug_aspect_span=getattr(cfg, "debug_aspect_span", False),
    )
    
    test_set = AspectSentimentDataset(
        json_path=cfg.test_path,
        tokenizer=tokenizer,
        max_len_sent=cfg.max_len_sent,
        max_len_term=cfg.max_len_term,
        label2id=train_set.label2id,
        mode=getattr(cfg, "mode", None),
        debug_aspect_span=getattr(cfg, "debug_aspect_span", False),
    )  

    return train_set, test_set 

def get_dataloader(cfg, train_set=None, val_set=None, test_set=None):
    
    train_loader, val_loader, test_loader = None, None, None
    
    if train_set is not None:
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
    
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
    
    if test_set is not None:
        test_loader = DataLoader(
            test_set,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader

def get_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)    
    return tokenizer

def set_seed(seed: int = 13):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    
