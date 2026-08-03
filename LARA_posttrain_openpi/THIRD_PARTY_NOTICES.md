# Third-party notices

LARA post-training is derived from
[Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi)
under the Apache License 2.0. The local history was based on OpenPI commit
`981483dca0fd9acba698fea00aa6e52d56a66c58`; subsequent LARA modifications
are recorded in this repository.

The latent motion tokenizer source is adapted from
[TencentARC/Moto](https://github.com/TencentARC/Moto), also released under
Apache License 2.0 with additional third-party notices in its
[LICENSE.txt](https://github.com/TencentARC/Moto/blob/main/LICENSE.txt).
The retained modules build on Transformers/ViT-MAE, PyTorch, torchvision, and
LPIPS; their upstream licenses remain authoritative.

The LIBERO submodule is distributed under the MIT License. Its dataset is
published separately under CC BY 4.0. See `third_party/libero/LICENSE` and
the LIBERO project documentation.

Gemma components and weights are subject to the Gemma Terms of Use included in
`LICENSE_GEMMA.txt`. The required distribution notice is included in
`NOTICE`.

This repository does not redistribute model checkpoints or datasets. Users
must review the terms attached to every downloaded checkpoint and dataset.
