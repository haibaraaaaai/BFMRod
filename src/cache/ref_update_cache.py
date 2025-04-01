from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import numpy as np


@dataclass
class SegmentCacheEntry:
    updated_refs: List[Tuple[int, np.ndarray]]  # list of (start_idx, smoothed_ref)
    phase0: np.ndarray  # unwrapped phase indices (concatenated 0-based indices)


class RefUpdateCacheManager:
    def __init__(self):
        self.cache: Dict[Tuple[Tuple[Tuple[int, int], Tuple[int, int]], float, float, float], SegmentCacheEntry] = {}

    def _make_key(
        self,
        ref_bounds: Tuple[Tuple[int, int], Tuple[int, int]],
        update_interval: float,
        alpha: float,
        fraction: float
    ) -> Tuple[Tuple[Tuple[int, int], Tuple[int, int]], float, float, float]:
        ref1 = tuple(map(int, ref_bounds[0]))
        ref2 = tuple(map(int, ref_bounds[1]))
        return (ref1, ref2), float(update_interval), float(alpha), float(fraction)

    def add_entry(
        self,
        ref_bounds: Tuple[Tuple[int, int], Tuple[int, int]],
        update_interval: float,
        alpha: float,
        fraction: float,
        entry: SegmentCacheEntry,
    ):
        key = self._make_key(ref_bounds, update_interval, alpha, fraction)
        self.cache[key] = entry

    def get_entry(
        self,
        ref_bounds: Tuple[Tuple[int, int], Tuple[int, int]],
        update_interval: float,
        alpha: float,
        fraction: float,
    ) -> Optional[SegmentCacheEntry]:
        key = self._make_key(ref_bounds, update_interval, alpha, fraction)
        return self.cache.get(key)

    def find_matching_pairs(
        self,
        computed_bounds: List[Tuple[int, int]],
        update_interval: float,
        alpha: float,
        fraction: float,
    ) -> List[SegmentCacheEntry]:
        """
        Search for all consecutive computed_ref_bound pairs that have cached entries.
        Returns a list of SegmentCacheEntry sorted in order.
        """
        entries = []
        for i in range(len(computed_bounds) - 1):
            pair = (computed_bounds[i], computed_bounds[i + 1])
            key = self._make_key(pair, update_interval, alpha, fraction)
            if key in self.cache:
                entries.append(self.cache[key])
        return entries
