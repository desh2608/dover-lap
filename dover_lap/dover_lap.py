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

from typing import List

from dover_lap.libs.rttm import load_rttm, write_rttm
from dover_lap.libs.turn import merge_turns, trim_turns, Turn
from dover_lap.libs.uem import load_uem
from dover_lap.libs.utils import error, info, warn, groupby, \
    command_required_option, PythonLiteralOption

from dover_lap.src.doverlap import DOVERLap


def load_rttms(
    rttm_list: List[str]
) -> List[List[Turn]]:
    """Loads speaker turns from input RTTMs in a list of turns."""
    turns_list = []
    file_ids = []
    for rttm_fn in rttm_list:
        try:
            turns_, _, file_ids_ = load_rttm(rttm_fn)
            turns_list.append(turns_)
            file_ids += list(file_ids_)
        except IOError as e:
            error("Invalid RTTM file: %s. %s" % (rttm_fn, e))
            sys.exit(1)
    return turns_list


@click.argument('input_rttms', nargs=-1, type=click.Path(exists=True))
@click.argument('output_rttm', nargs=1, type=click.Path())
@click.option('-u', '--uem-file', type=click.Path(), help='UEM file path')
@click.option('-c', '--channel', type=int, default=1, show_default=True,
            help='Use this value for output channel IDs')
@click.option('--second-maximal', is_flag=True, default=False, show_default=True,
            help='If this flag is set, run a second iteration of the maximal matching for'
                ' label mapping. It may give better results sometimes.')
@click.option('--tie-breaking', type=click.Choice(['uniform','all']), default='all',
            help='Specify whether to assign tied regions to all speakers or divide uniformly', show_default=True)
@click.option('--weight-type', type=click.Choice(['rank','custom']), default='rank',
            help='Specify whether to use rank weighting or provide custom weights', show_default=True)
@click.option('--dover-weight', type=float, default=0.1, help='DOVER weighting factor', show_default=True)
@click.option('--custom-weight', cls=PythonLiteralOption, help='Weights for input RTTMs')
@click.command(cls=command_required_option('weight_type',
                    {'custom':'custom_weight','rank':'dover_weight'}))
def main(
    input_rttms: List[click.Path],
    output_rttm: click.Path,
    uem_file: click.Path,
    channel: int,
    **kwargs # these are passed directly to combine_turns_list() method
) -> None:
    """Apply the DOVER-Lap algorithm on the input RTTM files."""
    
    # Load hypothesis speaker turns.
    info("Loading speaker turns from input RTTMs...", file=sys.stderr)
    turns_list = load_rttms(input_rttms)

    if uem_file is not None:
        info("Loading universal evaluation map...", file=sys.stderr)
        uem = load_uem(uem_file)

        # Trim turns to UEM scoring regions and merge any that overlap.
        info("Trimming reference speaker turns to UEM scoring regions...", file=sys.stderr)
        turns_list = [
            trim_turns(turns, uem) for turns in turns_list
        ]
            
    
    info("Merging overlapping speaker turns...", file=sys.stderr)
    turns_list = [
        merge_turns(turns) for turns in turns_list
    ]

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
        file_to_out_turns[file_id] = DOVERLap.combine_turns_list(
            turns_list,
            file_id,
            **kwargs
        )

    # Write output RTTM file
    write_rttm(
        output_rttm,
        sum(list(file_to_out_turns.values()), []),
        channel=channel
    )
