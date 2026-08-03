from typing import List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import rankdata

from dover_lap.libs.turn import Turn


class WeightedAverageVoting:
    def __init__(
        self,
        gaussian_filter_std: float = 0.01,
        cardinality_method: str = "mean",
        tie_method: str = "all",
    ) -> None:
        if cardinality_method not in {"mean", "median"}:
            raise ValueError(
                "Unknown cardinality method: {}".format(cardinality_method)
            )
        if tie_method not in {"all", "global"}:
            raise ValueError("Unknown tie method: {}".format(tie_method))
        self.gaussian_filter_std = gaussian_filter_std
        self.cardinality_method = cardinality_method
        self.tie_method = tie_method

    @staticmethod
    def _weighted_median(values, weights):
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        sorted_weights = weights[order]
        index = np.searchsorted(
            np.cumsum(sorted_weights),
            0.5 * sorted_weights.sum(),
            side="left",
        )
        return sorted_values[min(index, len(sorted_values) - 1)]

    def _get_num_speakers(self, raw_regions, smoothed_regions, weights):
        if self.cardinality_method == "mean":
            return np.sum(smoothed_regions, axis=1).round().astype(int)

        counts = np.count_nonzero(raw_regions > 0, axis=1)
        return np.asarray(
            [self._weighted_median(row, weights) for row in counts],
            dtype=int,
        )

    @staticmethod
    def _select_global(scores, num_speakers, global_scores):
        order = sorted(
            range(len(scores)),
            key=lambda speaker: (
                -scores[speaker],
                -global_scores[speaker],
                speaker,
            ),
        )
        return set(order[:num_speakers])

    def get_combined_turns(
        self,
        regions: np.ndarray,
        start_end: np.ndarray,
        file_id: str,
        weights: Optional[np.ndarray] = None,
    ) -> List[Turn]:
        """Combine mapped regions using mean or median speaker cardinality.

        With ``tie_method=global``, duration-weighted recording support is used
        only as a secondary score after the local activity vote.
        """
        assert (
            regions.shape[0] == start_end.shape[0]
        ), "Regions and start_end must have the same number of rows"

        if weights is None:
            weights = np.ones(regions.shape[2], dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        weights /= weights.sum()

        raw_regions = regions
        regions = np.sum(raw_regions, axis=2)
        regions = gaussian_filter1d(
            regions,
            sigma=self.gaussian_filter_std,
            axis=0,
            mode="nearest",
        )

        num_speakers = self._get_num_speakers(raw_regions, regions, weights)
        num_speakers = np.clip(num_speakers, 0, regions.shape[1])
        speaker_ranks = rankdata(-regions, axis=1, method="min")

        global_scores = np.zeros(regions.shape[1], dtype=np.float64)
        if self.tie_method == "global":
            durations = start_end[:, 1] - start_end[:, 0]
            global_scores = np.sum(regions * durations[:, None], axis=0)

        combined_turns = []
        for index, ranks in enumerate(speaker_ranks):
            if self.tie_method == "all":
                selected_speakers = {
                    speaker
                    for speaker, rank in enumerate(ranks)
                    if rank <= num_speakers[index]
                }
            else:
                selected_speakers = self._select_global(
                    regions[index], num_speakers[index], global_scores
                )

            start_time, end_time = start_end[index]
            for speaker in selected_speakers:
                if regions[index, speaker] > 0:
                    combined_turns.append(
                        Turn(
                            onset=start_time,
                            offset=end_time,
                            speaker_id=speaker,
                            file_id=file_id,
                        )
                    )

        return combined_turns
