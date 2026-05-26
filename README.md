# CIPT: Causal Interventional Prompt Tuning

## Install

```bash
pip install -r requirements.txt
```


## Quick Start

```python
import torch
from cipt import build_cipt, cipt_loss

classnames = ["cat", "dog", "bird"]
model, preprocess = build_cipt(
    classnames,
    backbone="ViT-B/16",
    num_diverse_templates=4,
)

images = torch.randn(8, 3, 224, 224).cuda()
labels = torch.randint(0, len(classnames), (8,), device="cuda")

out = model(images, labels)
losses = cipt_loss(
    out.interventional_logits,
    out.causal_logits,
    out.spurious_logits,
    out.causal_features,
    out.spurious_features,
    labels,
    beta=2.0,
    gamma=5.0,
)
losses.loss.backward()
```

## ImageFolder 

目录结构：

```text
data/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_a/*.jpg
    class_b/*.jpg
```

run：

```bash
python examples/train_cipt_imagefolder.py --data-root data --k 4 --beta 2 --gamma 5
```

settings：

- base-to-new: `beta=2`, `gamma=5`, `K=4`
- ImageNet: `beta=2`, `gamma=5`, `K=6`
- domain generalization: `beta=4`, `gamma=5`, `K=4`

