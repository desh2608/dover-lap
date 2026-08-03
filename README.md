# DOVER-Lap
Official implementation for [DOVER-Lap: A method for combining overlap-aware diarization outputs](https://arxiv.org/pdf/2011.01997.pdf).

## Installation

```shell
pip install dover-lap
```

## How to run

After installation, run

```shell
dover-lap [OPTIONS] OUTPUT_RTTM [INPUT_RTTMS]...
```

Example:

```shell
dover-lap egs/ami/rttm_dl_test egs/ami/rttm_test_*
```

For the robust, reference-free configuration developed on the bundled AMI
example, run:

```shell
dover-lap egs/ami/rttm_robust_test egs/ami/rttm_test_* \
  --label-mapping greedy-intersection \
  --voting-method median \
  --weight-type equal \
  --tie-method global \
  --gaussian-filter-std 0.01
```

This configuration uses raw co-active duration for label mapping, the median
input speaker count in each region, equal system weights, and recording-level
speaker support to break exact vote ties. It is training-free, runs entirely on
CPU, and does not use a reference RTTM during fusion. The original defaults are
unchanged.

## Usage instructions

```shell
Usage: dover-lap [OPTIONS] OUTPUT_RTTM [INPUT_RTTMS]...

  Apply the DOVER-Lap algorithm on the input RTTM files.

Options:
  --gaussian-filter-std FLOAT     Standard deviation for Gaussian filter
                                  applied before voting. This can help reduce
                                  the effect of outliers in the input RTTMs.
                                  For quick turn-taking, set this to a small
                                  value (e.g. 0.1). 0.5 is a good value for
                                  most cases. Set this to a very small value,
                                  e.g. 0.01, to remove filtering.  [default:
                                  0.5]

  --custom-weight TEXT            Weights for input RTTMs
  --dover-weight FLOAT            DOVER weighting factor  [default: 0.1]
  --weight-type [rank|custom|norm|equal]
                                  Specify whether to use rank weighting or
                                  provide custom weights  [default: rank]

  --tie-method [all|global]
                                  How to resolve a label-score tie at the
                                  cardinality cutoff  [default: all]

  --voting-method [average|median]
                                  Choose the local speaker-count estimator
                                  [default: average]

  --second-maximal                If this flag is set, run a second iteration
                                  of the maximal matching for greedy label
                                  mapping  [default: False]

  --label-mapping [hungarian|greedy|greedy-intersection]
                                  Choose label mapping algorithm to use
                                  [default: greedy]

  --random-seed INTEGER
  -c, --channel INTEGER           Use this value for output channel IDs
                                  [default: 1]

  -u, --uem-file PATH             UEM file path
  --help                          Show this message and exit.
```

**Note:** 

1. If `--weight-type custom` is used, then `--custom-weight` must be provided. For example:

```shell
dover-lap egs/ami/rttm_dl_test egs/ami/rttm_test_* --weight-type custom --custom-weight '[0.4,0.3,0.3]'
```

2. `label-mapping` can be set to `greedy` (default), `greedy-intersection`, or
`hungarian`. The Hungarian option is a modified version of the mapping technique
originally proposed in [DOVER](https://arxiv.org/abs/1909.08090).

## Results

We provide a sample result on the AMI mix-headset test set. The results can be 
obtained using [`spyder`](https://github.com/desh2608/spyder), which is automatically
installed with `dover-lap`:

```shell
dover-lap egs/ami/rttm_dl_test egs/ami/rttm_test_*
spyder egs/ami/ref_rttm_test egs/ami/rttm_dl_test
```

and similarly for the input hypothesis. The DER results are shown below.

|                                   |   MS  |  FA  | Conf. |  DER  |
|-----------------------------------|:-----:|:----:|:-----:|:-----:|
| Overlap-aware VB resegmentation   |  9.84 | **2.06** |  9.60 | 21.50 |
| Overlap-aware spectral clustering | 11.48 | 2.27 |  9.81 | 23.56 |
| Region Proposal Network           |  **9.49** | 7.68 |  8.25 | 25.43 |
| DOVER-Lap (Hungarian mapping)     | 9.98 | 2.13 | 8.25 | 20.35 |
| DOVER-Lap (Greedy mapping)*        | 9.96 | 2.16 |  7.75 | 19.86 |
| DOVER-Lap (Robust configuration)  | 10.27 | **1.72** | **6.99** | **18.99** |

_* The Greedy label mapping is exponential in number of inputs (see [this paper](https://arxiv.org/abs/2104.01954))._

The robust configuration was selected on the EN2002 and IS1009 meetings. On the
held-out ES2004 and TS3003 meetings it scores 14.56% DER, compared with 14.92%
for the default configuration. The 18.99% figure above is the final score over
all 17 bundled recordings.

## Running time

The algorithm is implemented in pure Python with NumPy for tensor computations. 
The time complexity is expected to increase exponentially with the number of 
inputs, but it should be reasonable for combining up to 10 input hypotheses. For 
combining more than 10 inputs, we recommend setting `--label-mapping hungarian`.

For smaller number of inputs (up to 5), the algorithm should take only a few seconds
to run on a laptop.

## Combining 2 systems with DOVER-Lap

DOVER-Lap is meant to be used to combine **more than 2 systems**, since
black-box voting between 2 systems does not make much sense. Still, if 2 systems
are provided as input, we fall back on the Hungarian algorithm for label mapping,
since it is provably optimal for this case. Both the systems are assigned equal
weights, and in case of voting conflicts, the region is assigned to both
labels. This is not the intended use case and will almost certainly lead
to performance degradation.

## Citation

```
@article{Raj2021Doverlap,
  title={{DOVER-Lap}: A Method for Combining Overlap-aware Diarization Outputs},
  author={D.Raj and P.Garcia and Z.Huang and S.Watanabe and D.Povey and A.Stolcke and S.Khudanpur},
  journal={2021 IEEE Spoken Language Technology Workshop (SLT)},
  year={2021}
}

@article{Raj2021ReformulatingDL,
  title={Reformulating {DOVER-Lap} Label Mapping as a Graph Partitioning Problem},
  author={Desh Raj and S. Khudanpur},
  journal={INTERSPEECH},
  year={2021},
}
```

## Contact

For issues/bug reports, please raise an Issue in this repository, or reach out to me at `draj@cs.jhu.edu`.
