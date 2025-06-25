# Conda environment setup

```
conda create --name diffcali python=3.10
conda activate diffcali
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install evotorch
```

You also need to install [NvDiffRast](https://nvlabs.github.io/nvdiffrast/) following the instructions below (for Linux):
```
git clone https://github.com/NVlabs/nvdiffrast
pip install .
pip install ninja
```

To install [Deep Hough Transform](https://github.com/Hanqer/deep-hough-transform), run
```
cd deep-hough-transform
cd model/_cdht
python setup.py build 
python setup.py install --user
pip install pot
```

# Calibration with differentiable rendering


Run:
```
python scripts/sequential_tracing.py --use_nvdiffrast --use_bbox_optimizer --tracking_visualization --final_iters 500 --online_iters 10
```


