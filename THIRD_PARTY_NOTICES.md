# Third-party notices

This repository is released under the BSD 3-Clause License (see [LICENSE](LICENSE)).
That license covers the code written for this project. It does **not** cover the
third-party components vendored into the tree, which remain under their own
licenses and are listed below.

---

## MMPretrain (OpenMMLab)

* Vendored at: `TMB_MPNN/TMB_MPNN_code/v2.1.0/mmpretrain/`, `TMB_MPNN/TMB_MPNN_code/v2.1.1/mmpretrain/`
* Also partially vendored at: `TMB_MPNN/TMB_MPNN_code/mmpretrain/`
* Version: 1.2.0
* Copyright: Copyright 2020 OpenMMLab. All rights reserved.
* License: Apache License 2.0 — see `TMB_MPNN/TMB_MPNN_code/v2.1.1/mmpretrain/LICENSE`
* Upstream: https://github.com/open-mmlab/mmpretrain

MMPretrain supplies the config, registry, and runner machinery that TMB_MPNN is
built on. Individual files under `mmpretrain/` carry their own copyright headers,
and some bundle further third-party code (including contributions credited to
ByteDance, Facebook/Meta, Microsoft, NVIDIA, Salesforce, Habana Labs, and other
authors). Those per-file headers are authoritative for the code they head and
have been left intact.

---

## OpenFold

* Vendored at: `TMB_MPNN/TMB_MPNN_code/v2.1.0/openfold_dependencies/`, `TMB_MPNN/TMB_MPNN_code/v2.1.1/openfold_dependencies/`
* Copyright: Copyright 2021 AlQuraishi Laboratory; Copyright 2021 DeepMind Technologies Limited
* License: Apache License 2.0 — see `TMB_MPNN/TMB_MPNN_code/v2.1.1/openfold_dependencies/LICENSE`
* Upstream: https://github.com/aqlaboratory/openfold

Provides rigid-frame utilities, residue constants, and feature transforms. Each
file retains its original Apache 2.0 header.

---

## Notes on the Apache License 2.0 components

Both components above are Apache 2.0. If you redistribute this repository or a
derivative of it, Apache 2.0 Section 4 requires that you retain the license
text, the copyright notices, and the per-file attribution headers shipped here.
Modifications made for this project are not separately annotated in the vendored
files; treat the upstream versions linked above as the reference point for
diffing.
