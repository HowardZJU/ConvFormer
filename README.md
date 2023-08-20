# ConvFormer
The source code for our submission **"ConvFormer: Revisiting Transformer for Sequential Recommendations"**.


## Requirements
* Install Python(>=3.7), Pytorch(>=1.8), Tensorboard, Tensorflow, Pandas, Numpy. In particular, we use Python 3.9.13, Pytorch 1.12.1+cu116, Tensorboard 2.11.2, Tensorflow 1.14.1.
* If you plan to use GPU computation, install CUDA. We use CUDA 11.6 and CUDNN 8.0

## Overview
**ConvFormer** consists of stacked **LighTCN** layers to extract the user preference representation from user behavior logs, followed by a dot-product scaler for recommendation. 

## Datasets
- We reuse the datasets that are provided in [Google Drive](https://drive.google.com/drive/folders/1omfrWZiYwmj3eFpIpb-8O29wbt4SVGzP?usp=sharing)
 and [Baidu Netdisk](https://pan.baidu.com/s/1we2eJ_Vz9SM33PoRqPNijQ?pwd=kzq2). The downloaded dataset should be placed in the `data` folder.

## Reproduction

We have three approaches to reproduce the results in the main paper.
- Open the `./assets_anonymous` folder and check the `log.log` and tensorboard files in the corresponding model_data repo. For example, You can get the test performance of ConvFormer on the Yelp dataset via `cat assets/conv_beauty/log.log | grep Test`.
- In the `main.py` file, set the option `do_eval=True`, and load the corresponding `.pt` file in the `assets_anonymous` folder. In this way you can load the model trained in our environment, with training logs in the corresponding folder.
- Run the training pipeline from scratch by running `amlt run main_results.yaml main_results`, and pull the results by `amlt results main_results -o ./assets/`. You can also reproduce the full-sort performance with `amlt run full_results.yaml full_results`.

    - python src/main.py --model_name CONV --data_name Beauty --padding_mode 0 --conv_size 30 --full_sort 0 --batch_size 256 
    - python src/main.py --model_name CONV --data_name Sports_and_Outdoors --padding_mode 0 --conv_size 30 --full_sort 0 --batch_size 256 
    - python src/main.py --model_name CONV --data_name Toys_and_Games --padding_mode 0 --conv_size 30 --full_sort 0 --batch_size 256 
    - python src/main.py --model_name CONV --data_name Yelp --padding_mode 0 --conv_size 30 --full_sort 0 --batch_size 256 