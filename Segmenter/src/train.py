#!/usr/bin/env python3
"""
Fine-tuning XLM-RoBERTa for situation-entity segmentation as a BI
classification task, using a CRF output layer for structured prediction.
XLM-RoBERTa encoder → linear classifier → CRF (torchcrf).
Early stopping on B-EDU F1 (dev set), best checkpoint saved automatically

Usage (single GPU):
    python src/train.py \\
        --data_dir data/prepared_datasets/xlm-roberta-large \\
        --model    FacebookAI/xlm-roberta-large \\
        --epochs   10 --batch_size 64 --lr 3e-5

Inputs:
    data-dir/{train,dev,test}/  – HuggingFace Datasets produced by prepare_datasets.py

Outputs:
    output-dir/best_model.pt    – model weights of the best checkpoint
    output-dir/config.json      – run configuration
    output-dir/results_*.json   – final test metrics
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_from_disk
from nltk.metrics.segmentation import windowdiff
from torch import optim
from torchcrf import CRF
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


LABELS = {"B-EDU": 0, "I-EDU": 1, "O": 2}
BIO_TAGS = [tag for tag, _ in sorted(LABELS.items(), key=lambda x: x[1])]
NUM_TAGS = len(BIO_TAGS)
O_IDX    = LABELS["O"]

DEFAULT_MODEL = "FacebookAI/xlm-roberta-large"

# optional local model cache (avoids re-downloading on HPC clusters)
# set via: export XLM_HUB_ROOT="/path/to/cache"
HUB_ROOT = os.environ.get("XLM_HUB_ROOT")


def resolve_model_path(model_id: str) -> str:
    """
    Return a local filesystem path to the model if found in the HUB_ROOT cache
    directory, otherwise return model_id unchanged so the HuggingFace Hub is used
    as a fallback (automatic download).
    """
    if not HUB_ROOT:
        print(f"No local cache set, will use Hugging Face Hub: {model_id}")
        return model_id

    cache_subdir = "models--" + model_id.replace("/", "--")
    snapshots_dir = os.path.join(HUB_ROOT, cache_subdir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        print(f"Model not found in local cache, will use Hugging Face Hub: {model_id}")
        return model_id

    snaps = [
        os.path.join(snapshots_dir, s)
        for s in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, s))
    ]
    if snaps:
        latest = max(snaps, key=os.path.getmtime)
        print(f"Using local model cache: {latest}")
        return latest

    print(f"Local cache empty, will use Hugging Face Hub: {model_id}")

    return model_id


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)


def is_main() -> bool:
    # returns True if this process should perform main tasks like logging, checkpointing, etc. (if distributing is possible)
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


class XLMEduModel(torch.nn.Module):

    def __init__(self, pretrained: str):
        super().__init__()
        # XLM-RoBERTa encoder + linear classifier (mapping hidden states to tag logits) + CRF layer for structured prediction
        self.encoder    = AutoModel.from_pretrained(pretrained, output_hidden_states=False)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, NUM_TAGS)
        self.crf        = CRF(NUM_TAGS, batch_first=True)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # return emission scores (logits) for each token, shape = (batch, seq_len, num_tags)
        outputs = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"], # ensures padding tokens are ignored by encoder
        )
        # classification logits per token, emissions for CRF
        return self.classifier(outputs.last_hidden_state)

    def loss(
        self,
        emissions:      torch.Tensor,
        labels:         torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # CRF mask: True = valid tokens, False = padding tokens
        crf_mask   = attention_mask.bool()
        # CRF cannot handle ignore index -100, so map it to a valid tag (O)
        labels_crf = labels.clone()
        labels_crf[labels_crf == -100] = O_IDX
        
        # CRF implementation expects float32 even if model uses bf16/fp16
        emissions = emissions.float()

        # negative log-likelihood loss (negate because CRF returns log-likelihood)
        return -self.crf(
            emissions,
            labels_crf,
            mask=crf_mask,
            reduction="mean"
        )

    def decode(
        self,
        emissions:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[List[int]]:
        # ensures decoding ignores padding tokens
        crf_mask = attention_mask.bool()
        
        # best path decoding returns predicted tag sequences
        return self.crf.decode(
            emissions.float(),
            mask=crf_mask
        )


def compute_metrics(
    all_preds:  List[List[str]],
    all_labels: List[List[str]],
) -> Dict[str, float]:
    
    flat_preds  = [tag for seq in all_preds  for tag in seq]
    flat_labels = [tag for seq in all_labels for tag in seq]

    overall_p, overall_r, overall_f1, _ = precision_recall_fscore_support(
        flat_labels, flat_preds,
        labels=["B-EDU", "I-EDU"], average="weighted", zero_division=0,
    )

    (b_edu_p,), (b_edu_r,), (b_edu_f1,), _ = precision_recall_fscore_support(
        flat_labels, flat_preds,
        labels=["B-EDU"], average=None, zero_division=0,
    )

    accuracy = accuracy_score(flat_labels, flat_preds)

    exact_match = sum(
        pred_seq == label_seq
        for pred_seq, label_seq in zip(all_preds, all_labels)
    ) / max(1, len(all_labels))

    # total reference boundaries (B-EDU tokens) and average segment length
    total_boundaries = sum(t == "B-EDU" for t in flat_labels)
    avg_seg_length = len(flat_labels) / max(1, total_boundaries)
    # k = half the average reference segment length, controls tolerance for boundary shifts
    k = max(1, round(avg_seg_length / 2))

    wd_scores = []
    for pred_seq, label_seq in zip(all_preds, all_labels):
        # samples shorter than 2k+1 are skipped (windowdiff requirement to form at least one window)
        if len(label_seq) < 2 * k + 1:
            continue
        # convert token labels into binary boundary sequences
        ref_boundaries   = [t == "B-EDU" for t in label_seq]
        pred_boundaries  = [t == "B-EDU" for t in pred_seq]

        # how many segmentation boundaries differ within a sliding window of size k?
        wd_scores.append(
            windowdiff(
                ref_boundaries,
                pred_boundaries,
                k=k,
                boundary=True,
            )
        )

    # average across samples
    window_diff = np.mean(wd_scores) if wd_scores else "nan"

    return {
        "overall_precision": float(overall_p),
        "overall_recall":    float(overall_r),
        "overall_f1":        float(overall_f1),
        "b_edu_precision":   float(b_edu_p),
        "b_edu_recall":      float(b_edu_r),
        "b_edu_f1":          float(b_edu_f1),
        "accuracy":          float(accuracy),
        "exact_match":       float(exact_match),
        "window_diff":       float(window_diff), # lower is better
    }


def evaluate(
    model:    torch.nn.Module,
    loader:   DataLoader,
    device:   torch.device,
    use_fp16: bool,
) -> Dict[str, float]:
    
    model.eval()
    all_preds, all_labels = [], []
    
    # if model is wrapped in DDP/DataParallel, unwrap to access methods like decode
    core = model.module if isinstance(model, (DDP, torch.nn.DataParallel)) else model

    with torch.no_grad():
        for batch in loader:
            labels         = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            inputs = {
                "input_ids":      batch["input_ids"].to(device),
                "attention_mask": attention_mask,
            }

            # running encoder in lower precision, emissions will still be used by CRF, so dtype stability matters
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_fp16
            ):
                emissions = model(inputs)

            # CRF decoding, best tag sequence per sample
            decoded   = core.decode(emissions, attention_mask)
            labels_np = labels.cpu().numpy()

            for pred_seq, l_seq in zip(decoded, labels_np):
                # filter out padding/special tokens (-100); decoded and labels_np share the same batch dimension
                true_tags = [BIO_TAGS[int(l)] for l, p in zip(l_seq, pred_seq) if l != -100]
                pred_tags = [BIO_TAGS[int(p)] for l, p in zip(l_seq, pred_seq) if l != -100]

                all_preds.append(pred_tags)
                all_labels.append(true_tags)

    return compute_metrics(all_preds, all_labels)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",     
        default="data/prepared_datasets"
    )

    parser.add_argument(
        "--output-dir",   
        default=None,
        help="Directory for checkpoints and results. Defaults to runs/<model-slug> if not set."
    )

    parser.add_argument(
        "--model",        
        default=DEFAULT_MODEL,
        help="FacebookAI/xlm-roberta-large"
    )

    parser.add_argument(
        "--epochs",       
        type=int,   
        default=10
    )

    parser.add_argument(
        "--batch-size",   
        type=int,   
        default=64,
        help="Per-GPU batch size."
    )

    parser.add_argument(
        "--lr",           
        type=float, 
        default=3e-5
    )

    parser.add_argument(
        "--weight-decay", 
        type=float, 
        default=0.001
    )
    
    parser.add_argument(
        "--seed",         
        type=int,   
        default=42
    )

    parser.add_argument(
        "--patience",     
        type=int,   
        default=3,
        help="Early stopping patience (epochs without B-EDU F1 improvement)."
    )

    parser.add_argument(
        "--fp16",         
        action=argparse.BooleanOptionalAction, 
        default=True,
        help="BF16 mixed precision via torch.autocast (default: on). Disable with --no-fp16."
    )

    parser.add_argument(
        "--grad-accum",   
        type=int,   
        default=1,
        help="Gradient accumulation steps, Effective batch = batch_size x grad_accum."
    )

    parser.add_argument(
        "--local-rank",   
        type=int,
        default=int(-1),
        help="Local rank for distributed training."
    )

    args = parser.parse_args()

    # resolve model path (checks local cache, falls back to HF Hub when no local model found)
    model_path = resolve_model_path(args.model)
    model_slug = args.model.split("/")[-1]

    if args.output_dir is None:
        args.output_dir = os.path.join("runs", model_slug)

    # configure distributed training if launched with torchrun/torch.distributed.launch
    use_ddp = args.local_rank >= 0 and torch.cuda.device_count() > 1
    if use_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # enable mixed precision only on CUDA devices
    use_fp16 = args.fp16 and device.type == "cuda"

    set_seed(args.seed)

    # save config of main process
    if is_main():
        os.makedirs(args.output_dir, exist_ok=True)
        config = {
            **vars(args),
            "model_slug":     model_slug,
            "model_resolved": model_path,
        }
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"Model ID:          {args.model}")
        print(f"Model slug:        {model_slug}")
        print(f"Resolved path:     {model_path}")
        print(f"Output directory:  {args.output_dir}")
        print(f"Computing on:      {device}")
        if use_ddp:
            print(f"DistributedDataParallel across {dist.get_world_size()} GPUs")
        print(f"Mixed precision (BF16): {use_fp16}")
        print(f"Gradient accumulation:  {args.grad_accum} steps "
              f"(effective batch size: "
              f"{args.batch_size * args.grad_accum * (dist.get_world_size() if use_ddp else 1)})")


    if is_main():
        print(f"\nLoading datasets from {args.data_dir} ...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=True
    )
    collator  = DataCollatorForTokenClassification(
        tokenizer=tokenizer, 
        padding="longest", 
        label_pad_token_id=-100
    )

    # load preprocessed datasets (expects HuggingFace datasets)
    train_ds = load_from_disk(os.path.join(args.data_dir, "train"))
    dev_ds   = load_from_disk(os.path.join(args.data_dir, "dev"))
    test_ds  = load_from_disk(os.path.join(args.data_dir, "test"))

    if is_main():
        print(f"Examples: train={len(train_ds)}, dev={len(dev_ds)}, test={len(test_ds)}")

    #  create data loaders with optional distributed sampling
    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None

    train_loader  = DataLoader(
        train_ds, 
        args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collator,
    )

    dev_loader  = DataLoader(
        dev_ds, 
        args.batch_size, 
        shuffle=False, 
        collate_fn=collator
        )
    
    test_loader = DataLoader(
        test_ds, 
        args.batch_size, 
        shuffle=False, 
        collate_fn=collator
    )

    # model setup:
    model = XLMEduModel(model_path).to(device)

    if use_ddp: # wrap model for multi-GPU training with DistributedDataParallel (one process per GPU)
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)
    elif torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
        model = torch.nn.DataParallel(model)

    # reference to underlying model (unwrapped from DDP/DataParallel)
    core = model.module if isinstance(model, (DDP, torch.nn.DataParallel)) else model

    # optimizer setup
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        betas=(0.9, 0.999), 
        eps=1e-8, 
        weight_decay=args.weight_decay,
    )

    total_optimizer_steps = (len(train_loader) // args.grad_accum) * args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_optimizer_steps),
        num_training_steps=total_optimizer_steps,
    )

    # gradient scaler for mixed precision training
    scaler = torch.amp.GradScaler(enabled=use_fp16)


    # training loop setup:
    best_f1        = 0.0
    patience_count = 0
    ckpt_path      = os.path.join(args.output_dir, "best_model.pt")
    start          = time.time()

    for epoch in range(args.epochs):
        model.train()
        if use_ddp: # ensure different shuffling across epochs in distributed training
            train_sampler.set_epoch(epoch)
        optimizer.zero_grad()
        epoch_loss, optimizer_steps = 0.0, 0

        for i, batch in enumerate(tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs} [{model_slug}]",
            disable=not is_main(),
        )):
            labels         = batch.pop("labels").to(device)
            attention_mask = batch["attention_mask"].to(device)
            inputs = {
                "input_ids":      batch["input_ids"].to(device),
                "attention_mask": attention_mask,
            }

            # forward pass with mixed precision
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_fp16):
                emissions = model(inputs)
                loss      = core.loss(emissions, labels, attention_mask)
                loss      = loss / args.grad_accum

            # backward pass with gradient scaling
            scaler.scale(loss).backward()

            # optimizer step (only after accumulating enough gradients)
            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # check if optimizer step will be skipped due to inf/nan gradients
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()

                # only step scheduler if optimizer step wasn't skipped
                if scaler.get_scale() == scale_before:  # step wasn't skipped
                    scheduler.step()
                optimizer.zero_grad()
                optimizer_steps += 1

            epoch_loss += loss.item() * args.grad_accum


        # evaluation and checkpointing:
        if is_main():
            avg_loss = epoch_loss / max(1, len(train_loader))
            print(f"\nEpoch {epoch+1} [{model_slug}] | avg loss: {avg_loss:.4f} "
                  f"| optimizer steps: {optimizer_steps}")
            
            # evaluate on dev set for early stopping (based on B-EDU F1)
            scores = evaluate(model, dev_loader, device, use_fp16)

            for k, v in scores.items():
                flag = " (optimising)" if k == "b_edu_f1" else (
                       " (lower is better)" if k == "window_diff" else "")
                print(f"  {k:>20} : {v:.4f}{flag}")

            #  save checkpoint if B-EDU F1 improved
            if scores["b_edu_f1"] > best_f1:
                best_f1        = scores["b_edu_f1"]
                patience_count = 0
                torch.save(core.state_dict(), ckpt_path)
                print(f"  New best B-EDU F1: {best_f1:.4f} -> {ckpt_path}")
                stop_flag = torch.tensor(0, device=device)
            else:
                patience_count += 1
                print(f"  No improvement ({patience_count}/{args.patience})")
                stop_flag = torch.tensor(
                    1 if patience_count >= args.patience else 0, device=device
                )
        else:
            stop_flag = torch.tensor(0, device=device)

        # synchronize early-stopping decision so all ranks exit together
        if use_ddp:
            dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            if is_main():
                print(f"\nEarly stopping triggered after epoch {epoch+1}.")
            break


    # final evaluation on test set with best model:
    if is_main():
        print(f"\nTotal training duration: {time.time() - start:.1f}s")
        print(f"\nLoading best model (B-EDU F1: {best_f1:.4f})...")
        core.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

        test_scores = evaluate(model, test_loader, device, use_fp16)
        print("\nFinal test results:")
        for k, v in test_scores.items():
            flag = " (lower is better)" if k == "window_diff" else ""
            print(f"  {k:>20} : {v:.4f}{flag}")

        # save test results to json:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out_file = os.path.join(args.output_dir, f"results_{model_slug}_{ts}.json")
        with open(out_file, "w") as f:
            json.dump(
                {
                    "model":          args.model,
                    "model_resolved": model_path,
                    **{k: float(v) for k, v in test_scores.items()},
                },
                f, indent=2,
            )
        print(f"Results saved to {out_file}")

    # clean up distributed training
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
