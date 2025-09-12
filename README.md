# Separate Then Specialize: Stabilizing Nvdiffrec for PBR Texture Extraction for Shinny Surface

# Install

Please follow the same instruction to install the same environment as [Nvdiffrec](https://github.com/NVlabs/nvdiffrec).

**For RTX5090 user:**
1. Install Python3.12 in your environment
```bash
conda create -n dmodel python=3.12
activate dmodel
```
2. Install [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive)

3. Download the latest stable PyTorch version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

4. 
```bash
pip install ninja imageio PyOpenGL glfw xatlas gdown
pip install git+https://github.com/NVlabs/nvdiffrast/
imageio_download_bin freeimage
```
5. Download tiny-cuda-nn
```bash
export TCNN_CUDA_ARCHITECTURES=120
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```




# Overview:

**TL;DR:** This project is based on the [nvdiffrec](https://github.com/NVlabs/nvdiffrec) pipeline.  
I modified the material optimization by splitting it into two parts, each with different learning rates and loss functions.  
The goal is to recover textures more accurately without needing manual per-asset configurations.


# Issue Explain

In the Nvdiffrec pipeline, users need to set specific limits on metallic, roughness or albedo value, otherwise, the system may produce incorrect textures that cause noticeable errors in the final render when lighting conditions change.

Without specific settings such as increase supersampling value or manually set maximum value for certain channels, the optimization process can become unstable, failing to recover low roughness properties. Especially shiny surface contains dark color areas, Nvdiffrec can misclassify the black regions as high roughness and high metallic areas. 

<!-- 
Shiny surface has the characteristics that they has narrow and long specular. This sometimes can be treated as outliers and ignored by the network. It can also causing the material entanglement when comes to dark surface, where the network might misclassify it as metallic. The way to workaround is increase super sampling or iterations, and disable the metallic channel.  


Nvdiffrec pipeline requires manually set up per-asset configurations, such as disable certain material channel, or increase iterations. Without doing so, Nvdiffrec pipeline can experience unstable on recovering accurate texture map for shinny surface, especially on dark color with low roughness..-->

This behaviour likely comes from the fact that in the Nvdiffrec, all material properties: Albedo, Normal, Roughness, Metallic and optionally Occlusion, are predicted by one network, which can lead to entanglement between them.

![issue_duck](./images/issue_duck.jpg)
![issue_cow](./images/issue_cow.jpg)


# Our method
### Main Idea:
Inspired by the manual 3D asset creation workflow, we propose a “separate then specialize” strategy: The network first "focus on" the most visually dominant attribute (mesh and color), then progressively refine the reflectance properties.
![pipeline](./images/pipeline.jpg)

We separate the material optimization into two parts. One part predicts Albedo and Normal , while the other part predicts Roughness and Metallic. Each part is trained with a different learning rate, to reduce the entanglement that often occurs in the joint optimization all together. Moreover, each part is tailerd with specific weighted composite losses.

We combined two different loss functions for different MLP. In practical, the best performance was achieved with :
![loss](./images/loss_function.jpg)
Where Loss2 for MLP-B need to be outlier-sensitive loss function such as MSE. While others are outlier-robust loss functions, such as LogL1, SMAPE.



### Other Implementations:
**Logging:** <br>
Implemented TensorBoard logging to visualize losses, learning rates, and compare metrics more directly.

![logging](./images/tensorboard.jpg)

**Unpack textures:** <br>

Unpacked textures from the packed ORM map into separate occlusion, roughness, and metallic texture maps, making it easier to identify which channel is incorrect.

![unpack_textures](./images/unpack_textures.jpg)

# Result:


<img align="left" src="images/cow_compare.jpg" width="500px"/>


All experiments were ran for 1000 iterations, no super-sampling, only occlusion channel disabled (because Nvdiffrec pipeline only support direct lighting) . Generated texture maps are with resolution of 1024\*1024, and training renders used 512\*512. All tests ran on Linux machine with a NVIDIA RTX 5090. The loss function select for test on original method is relmse. 

Our method shows some improvements on recovering accurate textures. It less misclassify the dark color as high metallic and achieve a lower roughness value. If we also joint optimize the lighting, 

![duck_compare](./images/duck_compare.jpg)

Customized grey ball asset.

![grey_compare](./images/grey_compare.jpg)


**Turn on joint lighting optimization**

![jointLgt_1](./images/jointLgt_1.jpg)
![jointLgt_2](./images/jointLgt_2.jpg)



# Limitations & Future works:

### Training time
Training time is 2-3 times longer, while this could be mitigate by change the specifications for Multi-resolution hash grid positional encoding. By change the level from 16 to 4, it decrease the training time from 8.581 minutes to 4.688 minutes. While it decrease the details for the coarse stage, the details could be added back on the Finer stage. And it also help the mesh affect less by the "invisible holes" (but this should be discuss due to the direct lighting renderer).
![hash](./images/hash_grid_encoding.jpg)

### Mesh is worse


### Metric

Since our goal is recover the accurate textures, directly comparing the final render image with the reference cannot reflect the true outcome. For example on the left that the recover the shinny surface while the right oine doesn't not, however the metric is showing that the right one is better than the left one. Since we are using dataset that has ground truth textures, directly compare between textures would be ideal. However, auto uv unwrap generate different uv layout in each training, compare them on 2D space is not possible. Developing a method that can compare textures in 3D space is left for future work. 
![metric](./images/metric.jpg)

**Addition**

Extend from the Metric section, In order to support the dataset with ground truth textures in differentiable rendering pipeline, I propose a DCC-to-dataset protocol. We can export the scene data via USD and renders from 3D Digital Content Create application such as Houdini, and treat it like a real-world dataset, convert the USD data to LLFF format for pipeline that accept LLFF format (such as Instant-ngp). The converting script yo can found here. 



**Notes:** I tried to keep this page short and focuesd to reduce reading time. For full details, please ask me for the complete project report. Feedback and suggestions are very welcome, especially corrections if you notice any issues with my experiments.