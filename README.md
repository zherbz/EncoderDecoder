# Multi-step Reservoir Storage Volume Forecasting Using Deep Learning
We propose a new multi-step reservoir forecasting approach based on a hybrid residual CNN-LSTM deep learning Encoder-Decoder algorithm. 
The proposed approach offers a three-month, weekly averaged prediction of reservoir storage volume based on historical snow water equivalent (SWE) during the runoff season from April through June each year. 

# Network Architecture
![Architecture](https://github.com/zherbz/EncoderDecoder/blob/master/Architecture.png)

# Residual CNN Encoder
![ResCNN](https://github.com/zherbz/EncoderDecoder/blob/master/ResCNN.png)

# Dependencies
* Python 3
* Climata
* Tensorflow
* Matplotlib
* Sklearn
* Seaborn
* Datetime
* Pandas
* Numpy

# Site Map: Upper Stillwater Reservoir
This study focuses on the Upper Stillwater reservoir located at the top of the Central Utah Water Conservancy District’s (CUWCD) collection system in the Uinta Mountains. CUWCD is one of Utah's four largest specialty water districts that provides potable and secondary water to various water associations, conservancy districts, irrigation companies, and local residents. It spans eight counties with over \$3.5 billion in infrastructure. There are currently ten lakes/reservoirs maintained and operated by CUWCD that house non-potable water in excess of 1.6 million ac-ft. The storage levels for these reservoirs act as a barometer for the state’s water resources and provide insight for how to appropriately prepare for future water usage. The figure below shows Upper Stillwater (located in the middle of the figure) surrounded by a network of snow telemetry monitoring sites.
![Site Map](https://github.com/zherbz/EncoderDecoder/blob/master/Site%20Map.png)

# Training Target: Reservoir Storage Volume (ac-ft)
![SV](https://github.com/zherbz/EncoderDecoder/blob/master/SV.png)

# Training Features: Snow Water Equivalent (in)
![SWE](https://github.com/zherbz/EncoderDecoder/blob/master/SWE.png)
