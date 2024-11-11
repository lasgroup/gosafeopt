import torch
import gosafeopt
from gosafeopt.aquisitions.base_aquisition import BaseAquisition
from gosafeopt.tools.logger import Logger
from typing import Optional
from torch import Tensor


class SafeSet:
    current_safe_set = 0
    best_sage_set = 0
    y_min = -1e10
    global_y_min = -1e10
    i = 0
    safe_sets = []

    @classmethod
    def configure(cls, max_global_steps_without_progress: int, max_global_steps_without_progress_tolerance: int):
        cls.max_global_steps_without_progress = max_global_steps_without_progress
        cls.max_global_steps_without_progress_tolerance = max_global_steps_without_progress_tolerance

    @classmethod
    def get_current_safe_set(cls) -> Optional[Tensor]:
        if len(cls.safe_sets) == 0:
            return None
        else:
            return cls.safe_sets[cls.current_safe_set]

    @classmethod
    def update_safe_set(cls, X: Tensor, aquisition: BaseAquisition):
        new_safe_set = X[aquisition.safe_set(X)]  # New params considered safe
        safe_set = cls.get_current_safe_set()
        if safe_set is not None:
            # Remove parameters considered unsafe under the updated model
            still_safe = aquisition.safe_set(safe_set)
            cls.filter_safe_set(still_safe)  # Remove unsafe points
            cls.add_to_current_safe_set(new_safe_set)
        else:
            cls.add_new_safe_set(new_safe_set)

    @classmethod
    def filter_safe_set(cls, mask: Tensor):
        cls.safe_sets[cls.current_safe_set] = cls.safe_sets[cls.current_safe_set][mask]

    @classmethod
    def add_to_current_safe_set(cls, safeset: Tensor):
        safeset.to(gosafeopt.device)
        current_safe_set = cls.get_current_safe_set()
        if current_safe_set is not None:
            cls.safe_sets[cls.current_safe_set] = torch.vstack([current_safe_set, safeset])
        else:
            cls.safe_sets[cls.current_safe_set] = safeset

    @classmethod
    def add_new_safe_set(cls, safeset: Tensor):
        safeset.to(gosafeopt.device)
        cls.safe_sets.append(safeset)

    @classmethod
    def change_to_latest_safe_set(cls):
        cls.i = 0
        cls.current_safe_set = len(cls.safe_sets) - 1
        cls.y_min = -1e10
        Logger.info(f"BestSet: {cls.best_sage_set} / CurrentSet: {cls.current_safe_set}")

    @classmethod
    def change_to_best_safe_set(cls):
        cls.i = 0
        cls.current_safe_set = cls.best_sage_set
        Logger.info(f"Changing to best set Nr. {cls.best_sage_set}")

    @classmethod
    def calculate_current_set(cls, yMin):
        if cls.global_y_min < yMin:
            cls.global_y_min = yMin
            cls.best_sage_set = cls.current_safe_set
            Logger.info(f"BestSet: {cls.best_sage_set}")

        if cls.y_min < yMin:
            cls.y_min = yMin

        if cls.y_min < (
            cls.max_global_steps_without_progress_tolerance * cls.global_y_min
            if cls.global_y_min > 0
            else (2 - cls.max_global_steps_without_progress_tolerance) * cls.global_y_min
        ):
            cls.i += 1

        if cls.i >= cls.max_global_steps_without_progress:
            cls.change_to_best_safe_set()
