# Hybrid Tracking Method (MixFormerV2 x KCF)

The algorithm uses a switch mechanism between two trackers: a fast Correlation Filter (KCF) and an accurate Visual Transformer (MixFormerV2), balancing tracking speed and quality.

### Overview

#### Algorithm

- The system uses **perceptual hashing** to compare the regions of interest (ROIs) from consecutive frames. 
- A high Hamming distance between hash codes indicates significant object movement, prompting a switch from the Correlation Filter to the Transformer.
- The Correlation Filter runs by default, and when it 'starts to lose' the object (determined by the Hamming distance exceeding a threshold), the system switches to the Transformer for more precise tracking.
- After the Transformer corrects the object’s position, the Correlation Filter is *reinitialized* using the updated bounding box, continuing the hybrid tracking process.

#### Results

- The proposed algorithm can play a crucial role in running on resource-constrained computers that are used on board UAVs or other robots. 
- Experiments have shown that the proposed algorithm can indeed significantly improve the tracking speed relative to current state-of-the-art models without significant loss of quality. 
- On the UAV123 and VisDrone-SOT datasets, it achieved a **59% and 119% speedup** respectively with only a **2-3% decrease in accuracy** compared to MixFormerV2 (experiments were conducted on Jetson Orin).

You can read more about the algorithm and results in the paper: *Sardaryan A., Sahakyan V., Melkonyan V., Sargsyan S. An Accurate Real-Time Object Tracking Method for Resource-Constrained Devices, Trudy ISP RAN/Proc. ISP RAS, vol. 36, issue 3, 2024. pp. 283-294 (in Russian). DOI: 10.15514/ISPRAS-2024-36(3)-20.*

### Run the code

This version is designed to run the tracker on the video received from the camera.

1. set paths in CmakeLists.txt
2. translate model from onnx format to engine: `/usr/src/tensorrt/bin/trtexec --onnx=onnx_model_path --saveEngine=save_engine_model_path`
3. connect the camera through usb (`/dev/video0`)
4. mkdir build && cd build
5. cmake ..
6. make
7. ./tracker [modelPath] [bitsThreshold] 2>nul
