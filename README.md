# Unitree G1 Development Applications

## Cloning

After cloning, don't forget to run:
```
git submodule update --init --recursive
```

## Configuration
- Install miniforge3 on unitree (aarch64 conda equivalent)
- Create a unitree conda environment with python 3.8
    - all further package configuration should be contained in your env
- Follow avp_teleoperate README
    - Note: nlopt version in requirements.txt may cause error. Manually installing most recent/compatible version of nlopt worked fine
- Follow unitree_sdk2_python README
    - Note: extra setup configuring the CycloneDDS library may be necessary
- Follow sam README
    - mkdir sam/checkpoints and download the sam_vit_b_01ec64.pth checkpoint into the folder
    - installing pytorch for jetson: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html (should be version torch-2.0.0+nv23.05)
    - installing torchvision for jetson: https://forums.developer.nvidia.com/t/install-torchvision-on-jetson-orin/220956 (should be version 0.15, installed from source)
