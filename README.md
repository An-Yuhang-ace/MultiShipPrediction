# Multi-Ship Trajectory Prediction Based on Social-LSTM  

## Project details

基于Social-LSTM的多船舶协同轨迹预测模型实现，本项目使用PyTorch构建了Social-LSTM模型，进行实验验证了考虑船舶间相互影响的Social-LSTM轨迹预测模型相比于普通LSTM模型的预测准确性。  

PyTorch implementation for Social-LSTM, which is built to predict multi-vessel trajectories. Experiments have been done to demonstrate that Social-LSTM can predict better trajectories than LSTM.

This project has been forked initially from <https://github.com/quancore/social-lstm>. If you find this code useful in your research, please cite the paper [CVPR16_Social_LSTM](http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf).

## Documentation

1. criterion.py: Python script for loss functions.  
Including a GaussianLikehood loss function and a RMSE loss function.
2. utils.py: Python script for handling input train/test/validation data and preprocess it.  
DataLoader class includes time_preprocess function, data load function, batch function and other data process function.  
3. model.py: Python script includes Social-LSTM and Vanilla-LSTM.  
Social-LSTM model implementation, Vanilla-LSTM model implementation and related functions.
4. helper.py: Python script includes various helper methods.  
5. train.py: Python script for training Social-LSTM model.
6. train_vlstm.py: Python script for training Vanilla-LSTM model.
7. test.py: Python script for model testing and getting output txt file for submission.
8. validation.py: Python script for externally evaluate a trained model by getting validation error.
9. visualize.py: Python script for visualizing predicted trajectories during train/test/validation sessions.  

## Results

|  Model  | Neighbor Size | Mean Error| Final Error|
|  :---:  | :-----------: | :-------: | :----: |
| Vanilla-LSTM | 0 |  0.6430 | 2.0371 |
| Social-LSTM |0.021| **0.6323** | 2.0572 |
| Social-LSTM |0.020| 0.6363 | **1.9084** |
| Social-LSTM |0.019| 0.6422 | 2.0148 |  
  
Subjective display of predicted trajectories  
![图片1](https://user-images.githubusercontent.com/34471199/155717370-83821f83-bea9-403d-a4f2-227eafdf3cac.png)
