# RetiFluidNet
RetiFluidNet: A Self-Adaptive and Multi-Attention Deep Convolutional Network for Retinal OCT Fluid Segmentation

Reza Rasti, Armin Biglari, Mohammad Rezapourian, Ziyun Yang, and Sina Farsiu
Recently accepted by IEEE Transactions on Medical Imaging.
Paper at: https://ieeexplore.ieee.org/document/9980422
________________________________________
Requirement: Tensorflow 2.4

This code including four parts:
1.	Codes for Reading data (/DataReader)
2.	Codes for RetiFluidNet model (/models)
3.	Codes for losses (/losses)
4.	Evaluation Code (/results)
Run the RetiFluidNet on your own network. 
If you want to run and compile the RetiFluidNet based on your own network, there are four simple steps:
1.	Replace Data paths into train.py
2.	Compile and run the train.py
Note: If you want to train the network, after replacing the pathes, just run the train.py
Reproduce the results in the paper (/paper_result)
For reproducing the result just run the /results.py
The pretrained models can be downloaded from Google Drive

Citation

If you find this work useful in your research, please consider citing:
“R. Rasti, A. Biglari, M. Rezapourian, Z. Yang and S. Farsiu, "RetiFluidNet: A Self-Adaptive and Multi-Attention Deep Convolutional Network for Retinal OCT Fluid Segmentation," in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2022.3228285.”

