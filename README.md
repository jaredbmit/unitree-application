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

## To do
- Re-do configuration instructions for python 3.10 (necessary for sam2)
