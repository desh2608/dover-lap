import numpy as np

from typing import List, Optional, Tuple

from dover_lap.libs.turn import Turn
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
    ) -> List[Turn]:
        """
        Implements combination using the DOVER-Lap label voting method.
        :param turns_list, list of mapped speaker turns (from each hypothesis)
        :param file_id, name of the file (recording/session)
        :param voting_method, which method to use for combining labels. Options:
            - `average`: use weighted average of labels
            - `hmm`: use a HMM-based voting method
        :param weights, hypothesis weights to use for rank weighting
        """
        regions, start_end = cls.__get_regions(turns_list, weights)

        if voting_method == "average":
            voter = WeightedAverageVoting()
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

        # Add weights to turns, and update speaker id
        if weights is None:
            weights = np.array([1] * len(turns_list))
        weighted_turns_list = []

        for turns, weight in zip(turns_list, weights):
            weighted_turns = [
                Turn(
                    turn.onset,
                    offset=turn.offset,
                    speaker_id=spk_index[turn.speaker_id],
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
            tokens.append(("START", turn.onset, turn.speaker_id, turn.weight))
            tokens.append(("END", turn.offset, turn.speaker_id, turn.weight))

        sorted_tokens = sorted(tokens, key=lambda x: (x[1], x[0]))

        regions_list = []
        region_start = sorted_tokens[0][1]
        running_speakers_dict = {sorted_tokens[0][2]: sorted_tokens[0][3]}
        for token in sorted_tokens[1:]:
            if token[1] - region_start > cls.EPS:
                running_speakers = [
                    (k, v) for k, v in running_speakers_dict.items() if v > 0
                ]
                if len(running_speakers) > 0:
                    regions_list.append(
                        (region_start, token[1], running_speakers.copy())
                    )
            if token[0] == "START":
                if token[2] in running_speakers_dict:
                    running_speakers_dict[token[2]] += token[3]
                else:
                    running_speakers_dict[token[2]] = token[3]
            else:
                running_speakers_dict[token[2]] -= token[3]
                if running_speakers_dict[token[2]] <= cls.EPS:
                    running_speakers_dict[token[2]] = 0
            region_start = token[1]

        # Build regions matrix and start_end matrix
        regions = np.zeros((len(regions_list), len(spk_index)), dtype=np.float32)
        start_end = np.zeros((len(regions_list), 2), dtype=np.float32)
        for i, region in enumerate(regions_list):
            start_end[i, 0] = region[0]
            start_end[i, 1] = region[1]
            for spk, weight in region[2]:
                regions[i, spk] = weight

        return regions, start_end
