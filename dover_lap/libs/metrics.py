"""Functions for computing DER.

Taken from https://github.com/nryant/dscore
"""
import os
import shutil
import subprocess
import tempfile

from .rttm import write_rttm

import numpy as np
from scipy.optimize import linear_sum_assignment


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
MDEVAL_BIN = os.path.join(SCRIPT_DIR, 'md-eval.pl')
def DER(ref_turns, sys_turns, collar=0.0, ignore_overlaps=False):
    """Return overall diarization error rate.
    Parameters
    ----------
    ref_turns : list of Turn
        Reference speaker turns.
    sys_turns : list of Turn
        System speaker turns.
    collar : float, optional
        Size of forgiveness collar in seconds. Diarization output will not be
        evaluated within +/- ``collar`` seconds of reference speaker
        boundaries.
        (Default: 0.0)
    ignore_overlaps : bool, optional
        If True, ignore regions in the reference diarization in which more
        than one speaker is speaking.
        (Default: False)
    Returns
    -------
    der : float
        Overall diarization error rate (in percent).
    References
    ----------
    NIST. (2009). The 2009 (RT-09) Rich Transcription Meeting Recognition
    Evaluation Plan. https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf
    """
    tmp_dir = tempfile.mkdtemp()

    # Write RTTMs.
    ref_rttm_fn = os.path.join(tmp_dir, 'ref.rttm')
    write_rttm(ref_rttm_fn, ref_turns)
    sys_rttm_fn = os.path.join(tmp_dir, 'sys.rttm')
    write_rttm(sys_rttm_fn, sys_turns)

    # Actually score.
    try:
        cmd = [MDEVAL_BIN,
               '-r', ref_rttm_fn,
               '-s', sys_rttm_fn,
               '-c', str(collar),
              ]
        if ignore_overlaps:
            cmd.append('-1')
        stdout = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        stdout = e.output
    finally:
        shutil.rmtree(tmp_dir)

    # Parse md-eval output to extract overall DER.
    stdout = stdout.decode('utf-8')
    for line in stdout.splitlines():
        if 'OVERALL SPEAKER DIARIZATION ERROR' in line:
            der = float(line.strip().split()[5])
            break
    return der
