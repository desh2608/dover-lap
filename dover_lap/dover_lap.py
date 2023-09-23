#!/usr/bin/env python3
"""
 This is the official implementation for the DOVER-Lap algorithm. It combines
 overlap-aware diarization hypotheses to produce an output RTTM.

 Raj, D., García-Perera, L.P., Huang, Z., Watanabe, S., Povey, D., Stolcke, A., & Khudanpur, S. 
 DOVER-Lap: A Method for Combining Overlap-aware Diarization Outputs.
 IEEE Spoken Language Technology Workshop 2021.

 Copyright  2020  Desh Raj (Johns Hopkins University)
"""
import sys
import click
import random
import numpy as np

from typing import List

from dover_lap.libs.rttm import load_rttm, write_rttm
from dover_lap.libs.turn import merge_turns, trim_turns, Turn
from dover_lap.libs.uem import load_uem
from dover_lap.libs.utils import (
    error,
    info,
    warn,
    groupby,
    command_required_option,
    PythonLiteralOption,
)

from dover_lap.src.doverlap import DOVERLap


def load_rttms(rttm_list: List[str]) -> List[List[Turn]]:
    """Loads speaker turns from input RTTMs in a list of turns."""
    turns_list = []
    file_ids = []
    for rttm_fn in sorted(rttm_list):
        try:
            turns_, _, file_ids_ = load_rttm(rttm_fn)
            turns_list.append(turns_)
            file_ids += list(file_ids_)
        except IOError as e:
            error("Invalid RTTM file: %s. %s" % (rttm_fn, e))
            sys.exit(1)
    return turns_list


@click.argument("input_rttms", nargs=-1, type=click.Path(exists=True))
@click.argument("output_rttm", nargs=1, type=click.Path())
@click.option("-u", "--uem-file", type=click.Path(), help="UEM file path")
@click.option(
    "-c",
    "--channel",
    type=int,
    default=1,
    show_default=True,
    help="Use this value for output channel IDs",
)
@click.option("--random-seed", type=int, default=0)
@click.option(
    "--label-mapping",
    type=click.Choice(["hungarian", "greedy"]),
    default="greedy",
    show_default=True,
    help="Choose label mapping algorithm to use",
)
@click.option(
    "--second-maximal",
    is_flag=True,
    default=False,
    show_default=True,
    help="If this flag is set, run a second iteration of the maximal matching for"
    " greedy label mapping",
)
@click.option(
    "--voting-method",
    type=click.Choice(["average"]),
    default="average",
    show_default=True,
    help="Choose voting method to use:"
    " average: use weighted average to combine input RTTMs",
)
@click.option(
    "--weight-type",
    type=click.Choice(["rank", "custom", "norm"]),
    default="rank",
    help="Specify whether to use rank weighting or provide custom weights",
    show_default=True,
)
@click.option(
    "--dover-weight",
    type=float,
    default=0.1,
    help="DOVER weighting factor",
    show_default=True,
)
@click.option(
    "--custom-weight", cls=PythonLiteralOption, help="Weights for input RTTMs"
)
@click.option(
    "--gaussian-filter-std",
    type=float,
    default=0.5,
    help="Standard deviation for Gaussian filter applied before voting. This can help"
    " reduce the effect of outliers in the input RTTMs. For quick turn-taking, set"
    " this to a small value (e.g. 0.1). 0.5 is a good value for most cases. Set this"
    " to a very small value, e.g. 0.01, to remove filtering.",
    show_default=True,
)
@click.command(
    cls=command_required_option(
        "weight_type", {"custom": "custom_weight", "rank": "dover_weight", "norm": None}
    )
)
def main(
    input_rttms: List[click.Path],
    output_rttm: click.Path,
    uem_file: click.Path,
    channel: int,
    random_seed: int,
    **kwargs,  # these are passed directly to combine_turns_list() method
) -> None:
    """Apply the DOVER-Lap algorithm on the input RTTM files."""

    # Set random seeds globally
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Load hypothesis speaker turns.
    info("Loading speaker turns from input RTTMs...", file=sys.stderr)
    turns_list = load_rttms(input_rttms)

    if uem_file is not None:
        info("Loading universal evaluation map...", file=sys.stderr)
        uem = load_uem(uem_file)

        # Trim turns to UEM scoring regions and merge any that overlap.
        info(
            "Trimming reference speaker turns to UEM scoring regions...",
            file=sys.stderr,
        )
        turns_list = [trim_turns(turns, uem) for turns in turns_list]

    info("Merging overlapping speaker turns...", file=sys.stderr)
    turns_list = [merge_turns(turns) for turns in turns_list]

    file_to_turns_list = dict()
    for turns in turns_list:
        for fid, g in groupby(turns, lambda x: x.file_id):
            if fid in file_to_turns_list:
                file_to_turns_list[fid].append(list(g))
            else:
                file_to_turns_list[fid] = [list(g)]

    # Run DOVER-Lap algorithm
    file_to_out_turns = dict()
    for file_id in file_to_turns_list:
        info("Processing file {}..".format(file_id), file=sys.stderr)
        turns_list = file_to_turns_list[file_id]
        random.shuffle(
            turns_list
        )  # We shuffle so that the hypothesis order is randomized
        file_to_out_turns[file_id] = DOVERLap.combine_turns_list(
            turns_list, file_id, **kwargs
        )

    # Write output RTTM file
    write_rttm(output_rttm, sum(list(file_to_out_turns.values()), []), channel=channel)
