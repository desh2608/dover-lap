from typing import List, Dict, Tuple

from dover_lap.libs.utils import groupby
from dover_lap.libs.turn import Turn


def compute_spk_intersection(
    ref_spk_turns: List[Turn],
    sys_spk_turns: List[Turn],
) -> float:
    """Compute the co-active duration of two speaker activity tracks."""
    ref_turns = sorted(ref_spk_turns, key=lambda turn: (turn.onset, turn.offset))
    sys_turns = sorted(sys_spk_turns, key=lambda turn: (turn.onset, turn.offset))

    intersection = 0.0
    ref_index = 0
    sys_index = 0
    while ref_index < len(ref_turns) and sys_index < len(sys_turns):
        ref_turn = ref_turns[ref_index]
        sys_turn = sys_turns[sys_index]
        intersection += max(
            0.0,
            min(ref_turn.offset, sys_turn.offset) - max(ref_turn.onset, sys_turn.onset),
        )
        if ref_turn.offset <= sys_turn.offset:
            ref_index += 1
        else:
            sys_index += 1

    return intersection


def compute_spk_overlap(ref_spk_turns: List[Turn], sys_spk_turns: List[Turn]) -> float:
    """
    Computes 'relative' overlap, i.e. Intersection Over Union
    """
    tokens = []
    all_turns = ref_spk_turns + sys_spk_turns
    for turn in all_turns:
        tokens.append(("BEG", turn.onset))
        tokens.append(("END", turn.offset))
    spk_count = 0
    ovl_duration = 0
    total_duration = 0
    for token in sorted(tokens, key=lambda x: x[1]):
        if token[0] == "BEG":
            spk_count += 1
            if spk_count == 2:
                ovl_begin = token[1]
            if spk_count == 1:
                speech_begin = token[1]
        else:
            spk_count -= 1
            if spk_count == 1:
                ovl_duration += token[1] - ovl_begin
            if spk_count == 0:
                total_duration += token[1] - speech_begin
    return ovl_duration / total_duration


def get_speaker_keys(turns_list: List[List[Turn]]) -> Dict[Tuple[int, int], str]:
    """
    Returns a dictionary which maps a file id (relative) and speaker id (relative)
    to absolute speaker id.
    """
    speakers_dict = {}
    for i, turns in enumerate(turns_list):
        turn_groups = {
            key: list(group) for key, group in groupby(turns, lambda x: x.speaker_id)
        }
        for j, key in enumerate(sorted(turn_groups.keys())):
            speakers_dict[(i, j)] = key
    return speakers_dict
