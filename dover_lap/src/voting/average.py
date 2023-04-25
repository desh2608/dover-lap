from typing import Optional, List

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import rankdata

from dover_lap.libs.turn import Turn


class WeightedAverageVoting:
    def __init__(self, gaussian_filter_std: float = 0.01) -> None:
        self.gaussian_filter_std = gaussian_filter_std

    def get_combined_turns(
        self, regions: np.ndarray, start_end: np.ndarray, file_id: str
    ) -> List[Turn]:
        """
        Implements combination using the DOVER-Lap weighted average voting method.

        :param regions, matrix of shape (num_regions, num_speakers, num_hypothesis).
            The value in cell (t, k, n) represents the weight speaker `k` in region `t`
            contributed by hypothesis `n`.
        :param start_end, list of start and end times for each region
        """
        assert (
            regions.shape[0] == start_end.shape[0]
        ), "Regions and start_end must have the same number of rows"

        # Sum the weights from all hypotheses
        regions = np.sum(regions, axis=2)

        # Apply Gaussian filter to the regions matrix along the T axis
        regions = gaussian_filter1d(
            regions, sigma=self.gaussian_filter_std, axis=0, mode="nearest"
        )

        # Get the number of speakers in each region
        num_spks = np.sum(regions, axis=1).round().astype(int)

        # Rank the weights in each region. We use the min method to break ties. This
        # means that if two speakers have the same weight, they will be assigned the
        # rank of the lower speaker. Note that we negate the regions matrix because
        # rankdata() sorts in ascending order.
        spk_ranks_matrix = rankdata(-1 * regions, axis=1, method="min")

        # Create turns list by combining the regions and start_end
        combined_turns_list = []
        for i in range(len(spk_ranks_matrix)):
            start_time, end_time = start_end[i]
            spk_ranks = spk_ranks_matrix[i]
            for j, spk_rank in enumerate(spk_ranks):
                if spk_rank <= num_spks[i]:
                    turn = Turn(
                        onset=start_time,
                        offset=end_time,
                        speaker_id=j,
                        file_id=file_id,
                    )
                    combined_turns_list.append(turn)

        return combined_turns_list
