## Welcome to the Implementation of "MoonSplat:Monocular Online Gaussian Splatting with Sim(3) Global Optimization" for SIGGRAPH 2026.

## 0.Setup
Create Conda Env
```
conda create -y -n MoonSplat python=3.10
conda activate MoonSplat

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

wget https://data.pyg.org/whl/torch-2.2.0%2Bcu118/torch_scatter-2.1.2%2Bpt22cu118-cp310-cp310-linux_x86_64.whl  
pip install ./torch_scatter-2.1.2+pt22cu118-cp310-cp310-linux_x86_64.whl

pip install numpy==1.26.4
pip install "lietorch @ git+https://github.com/princeton-vl/lietorch.git"

pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
pip install submodules/mast3r
pip install submodules/in3d
pip install .
```


## 1.Run
Our demo uses a sequence from ScannetV2 (http://www.scan-net.org/). Download and extract ScannetV2 frame using (https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py).

#### Make sure that "frames_path" in "config.yaml" are pointing to your downloaded data.
```
bash scripts/run.sh
```
Feel free to change hyperparameters in "config.yaml" to balance speed and performace. 

## 2.Check Results
In this demo, input frames looks like:
<div align="left">
  <img src="results/demo/frame-000000.color.jpg" alt="Reconstruction_Vis" width="300">
</div>

Visualizations during reconstruction:

<div align="left">
  <img src="results/demo/concat_frame_0002.png" alt="PCD_vis" width="600">
</div>

<div align="left">
  <img src="results/demo/concat_frame_0101.png" alt="PCD_vis" width="600">
</div>

<div align="left">
  <img src="results/demo/concat_frame_0124.png" alt="PCD_vis" width="600">
</div>

Final Reconstruction Video: "results/demo/slam_result_video.mp4"

<div align="left">
  <video width="600" controls>
    <source src="results/demo/slam_result_video.mp4" type="video/mp4">
  </video>
</div>

As our method is based on Scaffold-GS, please check Scaffold-GS (https://github.com/city-super/Scaffold-GS) for more visualization.

Feel free to Run for any other monocular sequences. Specify data path and normalized camera focal in 'config.yaml' and frame name pattern in 'dataloader.py'. 

As for now, camera focals are specified in "config.yaml". 
Our camera focal estimation module code are being organized and will be available soon. Before that, if camera focals are not provided by datasets, you can estimate camera focal from images using GeoCalib (https://github.com/cvg/GeoCalib).



## Thanks for your attention! More instructions will be added soon!
