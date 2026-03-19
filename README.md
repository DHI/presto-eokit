# presto-eokit

**presto-eokit** is a PyTorch-based library built on top of **Presto** (1) [Lightweight, Pre-trained Transformers for
Remote Sensing Timeseries] from NASA Harvest, with several enhancements:

* End-to-end, pipeline with one function `presto_eokit.generate_embeddings` for generating embeddings from different EO modalities (Sentinel-1, Sentinel-2, ERA5, SRTM and DW).
* `PrestoLightningModule` compatible with the expected Presto Datasets and Dataloaders for easy integration with PyTorch Lightning.
* Dataops, Model utilities and Optimizers for efficient experimentations.
* Easily process and encode EO data using Presto as feature extractor for downstream ML tasks.
* Similarity search using Presto embeddings and cosine similarity. 

## Installation 

Quick install from GitHub:

```bash
pip install "presto_eokit[cpu] @ git+https://github.com/DHI/presto-eokit.git"
# or
uv pip install "presto_eokit[cpu] @ git+https://github.com/DHI/presto-eokit.git"
```

**CPU/CUDA Flexibility**
Choose the right extra for your setup:

* `cpu`
* `cu124`, `cu126`, `cu128` — CUDA 12.4 / 12.6 / 12.8

Check your GPU with:

```bash
nvidia-smi
```
**Optional Extras**
You can also install feature-specific extras:

* `ml` – machine learning libraries
* `viz` – visualization libraries
* `dev` – developer tools (tests, linters, docs)
* `test` – testing tools
* `notebooks` – Jupyter support

---

<details>
<summary><strong>Option 1: <code>uv</code></strong></summary>

```bash
uv venv --python 3.12 
source .venv/bin/activate           # On Windows: .venv\Scripts\activate.bat

uv pip install "presto_eokit[cpu] @ git+https://github.com/DHI/presto-eokit.git"
# or for CUDA:
uv pip install "presto_eokit[cu124] @ git+https://github.com/DHI/presto-eokit.git"
```

Check device:

```bash
uv run python -c "import presto_eokit, torch; print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))"
```

</details>

---

<details>
<summary><strong>Option 2:  <code>mamba</code></strong></summary>

```bash
mamba create -n prestoeokit python=3.12 -y
mamba activate prestoeokit
```

Install package (choose **pip** or **uv pip**):

* **CPU**:

  ```bash
  pip install "presto_eokit[cpu] @ git+https://github.com/DHI/presto-eokit.git"
  # or
  uv pip install "presto_eokit[cpu] @ git+https://github.com/DHI/presto-eokit.git"
  ```

* **CUDA (e.g. 12.4)**:

  ```bash
  pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
      "presto_eokit[cu124] @ git+https://github.com/DHI/presto-eokit.git"
  # or
  uv pip install "presto_eokit[cu124] @ git+https://github.com/DHI/presto-eokit.git"
  ```

Check device:

```bash
python -c "import presto_eokit, torch; print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))"
```

</details>

---

<details>
<summary><strong>Option 3: Develop Locally</strong></summary>

```bash
uv init --python 3.12
uv add "presto_eokit[cpu] @ git+https://github.com/DHI/presto-eokit.git"
# or 
uv add "presto_eokit[cu124] @ git+https://github.com/DHI/presto-eokit.git"
```

</details>

---

## Usage


### Generate Embeddings

presto_eokit-Presto can be used to generate embeddings from multimodal EO data such as Sentinel-1, Sentinel-2, ERA5, SRTM and DW. Note that any combination of these modalities and bands is possible. These embeddings capture high level spatial and temporal representations in 128-dimensional feature vectors, which can be used for diverse downstream ML tasks.
- End-to-End pipline for generating Embeddings:[![View Jupyter Notebook](https://img.shields.io/badge/view-Jupyter%20notebook-lightgrey.svg)](https://github.com/DHI/presto-eokit/blob/main/notebooks/embeddings_E2E.ipynb)
```py
import torch
import rioxarray
import xarray as xr
import presto_eokit
from presto_eokit import Presto


device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
# Randomly initialized encoder-decoder model
#model = Presto.construct()
# Or to load pre-trained model 
model = Presto.load_pretrained(device)
model.to(device)
model.eval()  

da = xr.open_dataarray("/path/to/data") # .tif .nc 
s2_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"] 
da = da.assign_coords(band=s2_bands) 
band_config = {
  "s2": s2_bands,
  #"s1": ["VV", "VH"],
  #"era5": ["temperature_2m", "total_precipitation"],
  #"srtm": ["elevation", "slope"],
  }

embeds = presto_eokit.generate_embeddings(
    da=da,
    model=model,
    band_configs=band_config,
    month=6,
    timestep_dim=1,
    device="cpu",
    batch_size=1024,
    spatial_dims=("y", "x"),
    band_dim="band",
)
```

---

### Downstream Tasks

presto_eokit-Presto can be used as a feature extractor on top of any ML model such as RF, or as a trainable backbone for downstream EO and geospatial ML tasks. The setup is compatible with lightning and supports both regression and classification tasks. Training modes include linear probing or fine-tuning. 
The following example shows how to finetune Presto for a regression task.
```py
import torch
import lightning as L
from torchmetrics import R2Score, MeanSquaredError, MeanAbsoluteError
from presto_eokit import Presto, PrestoLightningModule
from presto_eokit.utils import set_seed


set_seed(42) 
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder_decoder = Presto.load_pretrained(device)
encoder_decoder.to(device)
encoder_decoder.eval() 

# downstream model
ft_model = encoder_decoder.construct_finetuning_model(num_outputs=1).to(device)
for param in ft_model.encoder.parameters():
    param.requires_grad = False

# training setup
metric_template = {
    "r2": R2Score(),
    "rmse": MeanSquaredError(squared=False),
    "mae": MeanAbsoluteError()
}
metrics = {split: {k: type(v)() for k, v in metric_template.items()} for split in ["train", "val", "test"]}
optimizer_config = {"base_lr": 1e-3, "weight_decay": 1e-4, "use_layerwise": False}
scheduler_config = {"mode": "min", "factor": 0.5, "patience": 5}
criterion = torch.nn.MSELoss()
pl_model = PrestoLightningModule(ft_model, criterion, optimizer_config, scheduler_config, metrics)

# train & evaluate
trainer = L.Trainer(max_epochs=100, accelerator="auto")
trainer.fit(pl_model, train_dataloaders=train_dl, val_dataloaders=val_dl)
trainer.test(pl_model, dataloaders=test_dl)
```

## Bibliography

1. Tseng, Gabriel, et al. "Lightweight, pre-trained transformers for remote sensing timeseries." arXiv preprint arXiv:2304.14065 (2023). https://doi.org/10.48550/arXiv.2304.14065