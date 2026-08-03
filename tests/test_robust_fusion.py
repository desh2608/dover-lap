import numpy as np
import pytest

from dover_lap.libs.turn import Turn
from dover_lap.src.mapping.map_utils import compute_spk_intersection
from dover_lap.src.voting import WeightedAverageVoting


def make_turn(start, end, speaker):
    return Turn(start, end, speaker_id=speaker, file_id="recording")


def active_speakers(turns, start, end):
    return {
        turn.speaker_id for turn in turns if turn.onset == start and turn.offset == end
    }


def test_median_cardinality_rejects_one_system_overlap_outlier():
    regions = np.zeros((1, 3, 3), dtype=np.float32)
    regions[0, 0, :] = 1.0 / 3.0
    regions[0, 1:, 0] = 1.0 / 3.0
    start_end = np.asarray([[0.0, 1.0]])
    weights = np.asarray([1.0 / 3.0] * 3)

    mean_voter = WeightedAverageVoting(cardinality_method="mean")
    median_voter = WeightedAverageVoting(cardinality_method="median")

    mean_turns = mean_voter.get_combined_turns(regions, start_end, "recording", weights)
    median_turns = median_voter.get_combined_turns(
        regions, start_end, "recording", weights
    )

    assert len(mean_turns) > len(median_turns)
    assert active_speakers(median_turns, 0.0, 1.0) == {0}


def test_global_tie_break_prefers_recording_level_support():
    regions = np.zeros((2, 2, 3), dtype=np.float32)
    regions[0, 0, 0] = 1.0 / 3.0
    regions[0, 1, 1] = 1.0 / 3.0
    regions[1, 1, :] = 1.0 / 3.0
    start_end = np.asarray([[0.0, 1.0], [1.0, 5.0]])
    weights = np.asarray([1.0 / 3.0] * 3)

    voter = WeightedAverageVoting(cardinality_method="median", tie_method="global")
    turns = voter.get_combined_turns(regions, start_end, "recording", weights)

    assert active_speakers(turns, 0.0, 1.0) == {1}


def test_speaker_intersection():
    left = [make_turn(0, 4, "left")]
    right = [make_turn(2, 6, "right")]
    assert compute_spk_intersection(left, right) == pytest.approx(2.0)
