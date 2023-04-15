from typing import Optional, List

import numpy as np
from scipy.stats import rankdata

from dover_lap.libs.turn import Turn


class WeightedAverageVoting:
    def __init__(self) -> None:
        pass

    def get_combined_turns(
        self, regions: np.ndarray, start_end: np.ndarray, file_id: str
    ) -> List[Turn]:
        """
        Implements combination using the DOVER-Lap weighted average voting method.

        :param regions, matrix of shape (num_regions, num_speakers), where each row
            represents a homogeneous time region, and each column represents a speaker.
            The value in each cell represents the weight of the speaker in that region.
        :param start_end, list of start and end times for each region
        """
        assert (
            regions.shape[0] == start_end.shape[0]
        ), "Regions and start_end must have the same number of rows"

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
