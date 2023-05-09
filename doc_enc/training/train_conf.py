#!/usr/bin/env python3

from enum import Enum
from typing import List, Optional, Dict
import dataclasses
from omegaconf import MISSING

from doc_enc.training.types import DocRetrLossType, TaskType, SentRetrLossType


class LRSchedulerKind(Enum):
    NONE = 0
    LINEAR = 1
    MULT = 2
    CYCLIC = 3
    ONE_CYCLE = 4


class OptimKind(Enum):
    SGD = 1
    ADAM = 2
    RADAM = 3
    NADAM = 4
    ADAMW = 5


@dataclasses.dataclass
class ParamGroupConf:
    lr: Optional[float] = None
    final_lr: Optional[float] = None
    weight_decay: Optional[float] = None
    momentum: Optional[float] = None

    max_grad_norm: Optional[float] = None


@dataclasses.dataclass
class OptimConf:
    max_grad_norm: float = 0.0

    optim_kind: OptimKind = OptimKind.ADAM
    weight_decay: float = 0.0
    # for SGD
    momentum: float = 0.9
    use_zero_optim: bool = False

    # LR
    lr: float = MISSING
    final_lr: float = 0.0
    emb: ParamGroupConf = ParamGroupConf()
    sent: ParamGroupConf = ParamGroupConf()
    sent_for_doc: ParamGroupConf = ParamGroupConf()
    sent_index: ParamGroupConf = ParamGroupConf()
    fragment: ParamGroupConf = ParamGroupConf()
    doc: ParamGroupConf = ParamGroupConf()
    doc_index: ParamGroupConf = ParamGroupConf()

    lr_scheduler: LRSchedulerKind = LRSchedulerKind.NONE
    lr_scheduler_kwargs: Dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TrainerConf:
    tasks: List[TaskType] = dataclasses.field(default_factory=list)
    eval_tasks: List[TaskType] = dataclasses.field(default_factory=list)

    optim: OptimConf = MISSING
    max_updates: int = MISSING

    sent_retr_loss_type: SentRetrLossType = SentRetrLossType.BICE
    doc_retr_loss_type: DocRetrLossType = DocRetrLossType.CE

    save_path: str = ''
    resume_checkpoint: str = ''
    restore_lr_scheduler_state: bool = True

    use_grad_checkpoint: bool = False
    emb_grad_scale: float = 0.0

    switch_tasks_every: int = 10
    log_every: int = 100
    eval_every: int = 300_000
    checkpoint_every: int = 200_000

    debug_iters: List[int] = dataclasses.field(default_factory=list)
    print_batches: bool = False
    print_gpu_memory_stat_every: int = 0
    spam_allocated_memory_info: bool = False
