# Beyond Linear Probes: Dynamic Safety Monitoring for Language Models


[![arXiv](https://img.shields.io/badge/arXiv-2509.26238-red)](https://arxiv.org/abs/2509.26238)

[**James Oldfield**](https://james-oldfield.github.io/)<sup>1</sup>, [**Philip Torr**](https://torrvision.com/)<sup>2</sup>, [**Ioannis Patras**](https://www.eecs.qmul.ac.uk/~ioannisp/)<sup>1</sup>, [**Adel Bibi**](https://www.adelbibi.com/)<sup>2</sup>
[**Fazl Barez**](https://fbarez.github.io/)<sup>2,3,4</sup>


<sup>1</sup>Queen Mary University of London, <sup>2</sup>University of Oxford, <sup>3</sup>WhiteBox, <sup>4</sup>Martian

> Monitoring large language models' (LLMs) activations is an effective way to detect harmful requests before they lead to unsafe outputs. However, traditional safety monitors often require the same amount of compute for every query. This creates a trade-off: expensive monitors waste resources on easy inputs, while cheap ones risk missing subtle cases. We argue that safety monitors should be flexible--costs should rise only when inputs are difficult to assess, or when more compute is available. To achieve this, we introduce **Truncated Polynomial Classifiers (TPCs)**, a natural extension of linear probes for dynamic activation monitoring. Our key insight is that polynomials can be trained and evaluated progressively, term-by-term. At test-time, one can early-stop for lightweight monitoring, or use more terms for stronger guardrails when needed. TPCs provide two modes of use. First, as a safety dial: by evaluating more terms, developers and regulators can "buy" stronger guardrails from the same model. Second, as an adaptive cascade: clear cases exit early after low-order checks, and higher-order guardrails are evaluated only for ambiguous inputs, reducing overall monitoring costs. On two large-scale safety datasets (WildGuardMix and BeaverTails), for 4 models with up to 30B parameters, we show that TPCs compete with or outperform MLP-based probe baselines of the same size, all the while being more interpretable than their black-box counterparts.


---

<img src="./figures/main.gif" width="1200px" height="auto">

## Overview

The codebase contains the following key files:

* `model.py` contains the model definitions (for the TPC and baselines)
* `train.py` contains the training scripts
* `test_poly_forward.py` contains unit tests to ensure that the symmetric forward pass matches that when materializing full tensors
* `utils.py` helper utils
* `extract/*` contains files to save intermediate activations to disk
* `sweep_monitors.py` is the main script to reproduce the results.
* `sweep.sh` is the main example script to train all models and reproduce the results.


## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@misc{oldfield2025tpc,
    title={Beyond Linear Probes: Dynamic Safety Monitoring for Language Models},
    author={James Oldfield and Philip Torr and Ioannis Patras and Adel Bibi and Fazl Barez},
    year={2025},
    eprint={2509.26238},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Contact

**Please feel free to get in touch at**: `jamesalexanderoldfield@gmail.com`