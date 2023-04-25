import copy
import numpy as np

from collections import defaultdict
from typing import List, Optional, Tuple

from dover_lap.libs.turn import Token, Turn
from dover_lap.src.voting import WeightedAverageVoting


class LabelVoting:
    EPS = 1e-3  # to avoid float equality check errors

    @classmethod
    def get_combined_turns(
        cls,
        turns_list: List[List[Turn]],
        file_id: str,
        voting_method: Optional[str] = "average",
        weights: Optional[np.array] = None,
        gaussian_filter_std: float = 0.01,
    ) -> List[Turn]:
        """
        Implements combination using the DOVER-Lap label voting method.
        :param turns_list, list of mapped speaker turns (from each hypothesis)
        :param file_id, name of the file (recording/session)
        :param voting_method, which method to use for combining labels. Options:
            - `average`: use weighted average of labels
            - `hmm`: use a HMM-based voting method
        :param weights, hypothesis weights to use for rank weighting
        :param gaussian_filter_std, standard deviation of Gaussian filter to apply
        :return: combined turns
        """
        regions, start_end = cls.__get_regions(turns_list, weights)

        if voting_method == "average":
            voter = WeightedAverageVoting(gaussian_filter_std=gaussian_filter_std)
        else:
            raise ValueError("Unknown voting method: {}".format(voting_method))

        combined_turns_list = voter.get_combined_turns(regions, start_end, file_id)

        return combined_turns_list

    @classmethod
    def __get_regions(
        cls, turns_list: List[List[Turn]], weights: Optional[np.array] = None
    ) -> List[Tuple[float, float, List[Tuple[int, float]]]]:
        """
        Returns homogeneous time regions of the input.
        """
        # Map speaker ids to consecutive integers (0, 1, 2, ...)
        spk_index = {}
        for turns in turns_list:
            for turn in turns:
                if turn.speaker_id not in spk_index:
                    spk_index[turn.speaker_id] = len(spk_index)

        # Add weights to turns, and update speaker id. New speaker id is a tuple of (hyp_index, spk_index)
        if weights is None:
            weights = np.array([1] * len(turns_list))
        weighted_turns_list = []

        for i, (turns, weight) in enumerate(zip(turns_list, weights)):
            weighted_turns = [
                Turn(
                    turn.onset,
                    offset=turn.offset,
                    speaker_id=(i, spk_index[turn.speaker_id]),
                    file_id=turn.file_id,
                    weight=weight,
                )
                for turn in turns
            ]
            weighted_turns_list.append(weighted_turns)
        all_turns = [turn for turns in weighted_turns_list for turn in turns]

        tokens = []
        for turn in all_turns:
            # Name is 'START' (not 'BEG') so that 'END' tokens come first for same timestamp
            tokens.append(
                Token(
                    type="START",
                    timestamp=turn.onset,
                    hyp_index=turn.speaker_id[0],
                    speaker=turn.speaker_id[1],
                    weight=turn.weight,
                )
            )
            tokens.append(
                Token(
                    type="END",
                    timestamp=turn.offset,
                    hyp_index=turn.speaker_id[0],
                    speaker=turn.speaker_id[1],
                    weight=turn.weight,
                )
            )

        sorted_tokens = sorted(tokens, key=lambda x: (x.timestamp, x.type))

        regions_list = []
        region_start = sorted_tokens[0].timestamp
        # We also maintain a running dictionary of speakers and their weights contributed
        # by each hypothesis. These weights are stored as a list indexed by hyp_index.
        running_speakers_dict = defaultdict(
            lambda: [0.0 for _ in range(len(turns_list))]
        )
        running_speakers_dict[sorted_tokens[0].speaker][
            sorted_tokens[0].hyp_index
        ] += sorted_tokens[0].weight

        for token in sorted_tokens[1:]:
            if token.timestamp - region_start > cls.EPS:
                running_speakers = []
                for k, v in running_speakers_dict.items():
                    if sum(v) > cls.EPS:
                        running_speakers.append((k, v))
                if len(running_speakers) > 0:
                    regions_list.append(
                        (region_start, token.timestamp, copy.deepcopy(running_speakers))
                    )
            # Update the weights list for the current speaker
            weights = running_speakers_dict[token.speaker]
            if token.type == "START":
                weights[token.hyp_index] += token.weight
            else:
                weights[token.hyp_index] -= token.weight
            running_speakers_dict[token.speaker] = weights

            region_start = token.timestamp

        # Build regions matrix and start_end matrix. Regions matrix is of shape
        # (num_regions, num_speakers, num_hypotheses). start_end matrix is of shape
        # (num_regions, 2) and contains the start and end times of each region.
        regions = np.zeros(
            (len(regions_list), len(spk_index), len(turns_list)), dtype=np.float32
        )
        start_end = np.zeros((len(regions_list), 2), dtype=np.float32)
        for i, region in enumerate(regions_list):
            start_end[i, 0] = region[0]
            start_end[i, 1] = region[1]
            for spk, weights in region[2]:
                regions[i, spk] = weights

        return regions, start_end
