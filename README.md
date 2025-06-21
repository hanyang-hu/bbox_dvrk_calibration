# Conda environment setup

```
conda create --name diffcali python=3.10
conda activate diffcali
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install evotorch
```

You also need to install NvDiffRast following the instructions [here](https://nvlabs.github.io/nvdiffrast/).

# Calibration with differentiable rendering


Run:
```
python scripts/sequential_tracing.py --use_nvdiffrast --use_bbox_optimizer --tracking_visualization --final_iters 500 --online_iters 20
```


