"""Functions for reading/writing and manipulating NIST un-partitioned
evaluation maps.

An un-partitioned evaluation map (UEM) specifies the time regions within each
file that will be scored.

Taken from https://github.com/nryant/dscore
"""
from collections import defaultdict
from collections.abc import MutableMapping

import os

from intervaltree import IntervalTree


class UEM(MutableMapping):
    """Un-partitioned evaluation map (UEM).

    A UEM defines a mapping from file ids to scoring regions.
    """

    def __init__(self, *args, **kwargs):
        super(UEM, self).__init__()
        self.update(*args, **kwargs)

    def __setitem__(self, fid, score_regions):
        # Validate types. Expects sequence of (onset, offset) pairs.
        invalid_type_msg = "Expected sequence of pairs. Received: %r (%s)." % (
            score_regions,
            type(score_regions),
        )
        try:
            score_regions = [tuple(region) for region in score_regions]
        except TypeError:
            raise TypeError(invalid_type_msg)
        for score_region in score_regions:
            if len(score_region) != 2:
                raise TypeError(invalid_type_msg)

        # Validate that the (onset, offset) pairs are valid: no negative
        # timestamps or negative durations.
        def _convert_to_float(score_region):
            onset, offset = score_region
            try:
                onset = float(onset)
                offset = float(offset)
            except ValueError:
                raise ValueError(
                    "Could not convert interval onset/offset to float: %s"
                    % repr(score_region)
                )
            if onset >= offset or onset < 0:
                raise ValueError(
                    'Invalid interval (%.3f, %.3f) for file "%s".'
                    % (onset, offset, fid)
                )
            return onset, offset

        score_regions = [_convert_to_float(region) for region in score_regions]

        # Merge overlaps. Use of intervaltree Incurs some additional overhead,
        # but pretty small compared to actual scoring.
        tree = IntervalTree.from_tuples(score_regions)
        tree.merge_overlaps()
        score_regions = [(intrvl.begin, intrvl.end) for intrvl in tree]

        self.__dict__[fid] = score_regions

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return "{}, UEM({})".format(super(UEM, self).__repr__(), self.__dict__)


def load_uem(uemf):
    """Load un-partitioned evaluation map from file in NIST format.

    The un-partitioned evaluation map (UEM) file format contains
    one record per line, each line consisting of NN space-delimited
    fields:

    - file id  --  file id
    - channel  --  channel (1-indexed)
    - onset  --  onset of evaluation region in seconds from beginning of file
    - offset  --  offset of evaluation region in seconds from beginning of
      file

    Lines beginning with semicolons are regarded as comments and ignored.

    Parameters
    ----------
    uemf : str
        Path to UEM file.

    Returns
    -------
    uem : UEM
        Evaluation map.
    """
    with open(uemf, "rb") as f:
        fid_to_score_regions = defaultdict(list)
        for line in f:
            if line.startswith(b";"):
                continue
            fields = line.decode("utf-8").strip().split()
            file_id = os.path.splitext(fields[0])[0]
            onset = float(fields[2])
            offset = float(fields[3])
            fid_to_score_regions[file_id].append((onset, offset))
    return UEM(fid_to_score_regions.items())
