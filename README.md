# Histogram-Equalization-CUDA

The aim of the project is to compare two different executions of histogram equalization made with parallel programming modes CUDA and OpenMP.

This repository reports the project developed with CUDA and the results of both applications compared. The OpenMP implementation is available at the following link:


The times of execution of the two modalities have been compared considering the following aspects:
- variation of number of threads
- variation of image dimensions

The project was developed with the operating system Windows an the IDE Visual Studio. Require the installation of OpenCV. For replicate the result download the project, open Visual Studio and compute the following steps:
1. Click on File -> Open -> Project/Solution
2. Select the .sln project file and click to Open
3. Configuration Manager -> Active solution configuration -> Release
4. Configuration Manager -> Active solution platform -> x64
5. Build Dependencies -> Build Customizations -> Tick on CUDA
6. Properties -> Tab VC++ Directories -> Include Directories -> 
   
   C:\opencv\build\include (if opencv is installed in disk C)
   C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.2\common\inc
6. Properties -> Tab CUDA C/C++ -> Target Machine Platform -> 64-bit
7. Properties -> Tab Linker -> Sub-Tab Input -> Additional Dependencies -> cudart.lib, opencv_world420.lib, opencv_world420d.lib
8. Go in OpenCV installation folder -> build-> x64 -> vc14 -> bin -> copy the files opencv_world420.dll, opencv_world420d.dll
9. Paste it in the ImageProcessing folder of the project

In the folder _report_ there are:
- Relation of the project
- File excel that reports all time results
- Powerpoint presentation of the project
