import numpy as np

from typing import List, Optional, Tuple

from dover_lap.libs.turn import Turn


class LabelVoting:
    EPS = 1e-3  # to avoid float equality check errors

    @classmethod
    def get_combined_turns(cls,
        turns_list: List[Turn],
        file_id: str,
        tie_breaking: Optional[str]='uniform',
        weights: Optional[np.array]=None
    ) -> List[Turn]:
        """
        Implements combination using the DOVER-Lap label voting method.
        :param turns_list, list of mapped speaker turns
        :param file_id, name of the file (recording/session)
        :param tie_breaking, which method to use for breaking ties. Options:
            - `uniform`: divide ties uniformly between speakers
            - `all`: assign region to all tied speakers
        :param weights, hypothesis weights to use for rank weighting
        """
        combined_turns_list = []
        regions = cls.__get_regions(turns_list, weights)

        for region in regions:
            # Remove problematic regions
            if np.abs(region[1] - region[0]) < cls.EPS:
                continue
            
            # Get a list of speakers in this region in decreasing order of their weights
            spk_weights = sorted(region[2], key=lambda x: x[1], reverse=True)

            # Estimate the number of speakers in this region. This is computing by
            # adding all the speaker weights in this region (which is simply equal
            # to the weighted mean of number of speakers in the hypotheses).
            num_spk = int(round(sum([spk_weights[1] for spk_weights in region[2]])))
            
            i = 0
            while i < num_spk and len(spk_weights) > 0:
                cur_weight = spk_weights[0][1]
                filtered_ids = [
                    spk_id
                    for (spk_id, weight) in spk_weights
                    if np.abs(weight - cur_weight) < cls.EPS
                ]

                for j, spk_id in enumerate(filtered_ids):
                    if tie_breaking == 'uniform':
                        dur = (region[1] - region[0]) / len(filtered_ids)
                        start_time = region[0] + j * dur
                        offset = region[0] + (j + 1) * dur
                    elif tie_breaking == 'all':
                        start_time = region[0]
                        offset = region[1]
                
                    turn = Turn(
                        start_time,
                        offset=offset,
                        speaker_id=spk_id,
                        file_id=file_id,
                    )
                    combined_turns_list.append(turn)
                    
                i += len(filtered_ids)
                spk_weights = spk_weights[
                    len(filtered_ids) :
                ]  # remove the spk ids that have been seen

        return combined_turns_list

    @classmethod
    def __get_regions(cls,
        turns_list: List[Turn],
        weights: Optional[np.array]=None
    ) -> List[Tuple[float,float,List[Tuple[int,float]]]]:
        """
        Returns homogeneous time regions of the inputs as list of tuples. Regions are
        demarcated by start or end times of speaker intervals in any of the input
        hypothesis. Each region tuple also contains a list of the speakers (along
        with their weights) in that region.
        """
        if weights is None:
            weights = np.array([1] * len(turns_list))
        weighted_turns_list = []

        for turns, weight in zip(turns_list, weights):
            weighted_turns = [
                Turn(
                    turn.onset,
                    offset=turn.offset,
                    speaker_id=turn.speaker_id,
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

        regions = []
        region_start = sorted_tokens[0][1]
        running_speakers_dict = {sorted_tokens[0][2]: sorted_tokens[0][3]}
        for token in sorted_tokens[1:]:
            if token[1] > region_start:
                running_speakers = [
                    (k, v) for k, v in running_speakers_dict.items() if v > 0
                ]
                if len(running_speakers) > 0:
                    regions.append((region_start, token[1], running_speakers.copy()))
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

        return regions
        