from __future__ import annotations

from prog_policies.base import BaseTask

from .boxoban import BoxobanTask

TASK_NAME_LIST = [
    "Boxoban",
]


def get_task_cls(task_cls_name: str):
    # Allow both the registry name ("Boxoban") and the class name.
    if task_cls_name in ("Boxoban", "BoxobanTask"):
        return BoxobanTask
    return globals().get(task_cls_name)
