# DOVER-Lap
Official implementation for [DOVER-Lap: A method for combining overlap-aware diarization outputs](https://arxiv.org/pdf/2011.01997.pdf).

## Installation

To install, simply run:

```
pip install dover-lap
```

## How to run

After installation, run

```
dover-lap [OPTIONS] OUTPUT_RTTM [INPUT_RTTMS]...
```

Example:

```
dover-lap egs/ami/rttm_dl_test egs/ami/rttm_test_*
```

## Usage instructions

```
Usage: dover-lap [OPTIONS] OUTPUT_RTTM [INPUT_RTTMS]...

  Apply the DOVER-Lap algorithm on the input RTTM files.

Options:
  --custom-weight TEXT          Weights for input RTTMs
  --dover-weight FLOAT          DOVER weighting factor  [default: 0.1]
  --weight-type [rank|custom]   Specify whether to use rank weighting or
                                provide custom weights  [default: rank]

  --tie-breaking [uniform|all]  Specify whether to assign tied regions to all
                                speakers or divide uniformly  [default: all]

  --second-maximal              If this flag is set, run a second iteration of
                                the maximal matching for label mapping. It may
                                give better results sometimes.  [default:
                                False]

  -c, --channel INTEGER         Use this value for output channel IDs
                                [default: 1]

  -u, --uem-file PATH           UEM file path
  --help                        Show this message and exit.
```

**Note:** If `--weight-type custom` is used, then `--custom-weight` must be provided.
For example:

```
dover-lap egs/ami/rttm_dl_test egs/ami/rttm_test_* --weight-type custom --custom-weight '[0.4,0.3,0.3]'
```

## Results

We provide a sample result on the AMI mix-headset test set. The results can be 
obtained as follows:

```
dover-lap egs/ami/rttm_dl_test egs/ami/rttm_test_*
md-eval.pl -r egs/ami/ref_rttm_test -s egs/ami/rttm_dl_test
```

and similarly for the input hypothesis. The DER results are shown below.

|                                   |   MS  |  FA  | Conf. |  DER  |
|-----------------------------------|:-----:|:----:|:-----:|:-----:|
| Overlap-aware VB resegmentation   |  9.84 | **2.06** |  9.60 | 21.50 |
| Overlap-aware spectral clustering | 11.48 | 2.27 |  9.81 | 23.56 |
| Region Proposal Network           |  **9.49** | 7.68 |  8.25 | 25.43 |
| DOVER-Lap                         | 9.71 | 3.00 |  **7.59** | **20.30** |

**Note:** A version of md-eval.pl can be found in `dover_lap/libs`.

## Running time

The algorithm is implemented in pure Python with NumPy for tensor computations. 
The time complexity is expected to increase exponentially with the number of 
inputs, but it should be reasonable for combining up to 10 input hypotheses.

For smaller number of inputs (up to 5), the algorithm should take only a few seconds
to run on a laptop.

## Combining 2 systems with DOVER-Lap

DOVER-Lap is meant to be used to combine **more than 2 systems**, since
black-box voting between 2 systems does not make much sense. Still, if 2 systems
are provided as input, we fall back on the Hungarian algorithm for label mapping,
since it is provably optimal for this case. Both the systems are assigned equal
weights, and in case of voting conflicts, the region is equally divided among the
two labels. This is not the intended use case and will almost certainly lead
to performance degradation.

## Citation

```
@article{Raj2021Doverlap,
  title={{DOVER-Lap}: A Method for Combining Overlap-aware Diarization Outputs},
  author={D.Raj and P.Garcia and Z.Huang and S.Watanabe and D.Povey and A.Stolcke and S.Khudanpur},
  journal={2021 IEEE Spoken Language Technology Workshop (SLT)},
  year={2021}
}
```

## Contact

For issues/bug reports, please raise an Issue in this repository, or reach out to me at `draj@cs.jhu.edu`.
