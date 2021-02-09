# Self-Supervised Learning for Time Series Pre-training

This repository is the [PyTorch](http://pytorch.org/) implementation for the paper "Self-Supervised Learning for Time Series Pre-training"
<!-- <img src='./figure/task_all2.png' align="center" width="700px"> -->

## Environment
- Linux 
- python 3.7
- cuda 10.1

## Requirements
- [PyTorch](http://pytorch.org/) 
- [easydict](https://pypi.org/project/easydict/)
- [tensorboardX](https://pypi.org/project/tensorboardX/)
- [tqdm](https://pypi.org/project/tqdm/)
- [torchvision](https://pypi.org/project/tqdm/)
- json 
- [pyprobar](https://pypi.org/project/pyprobar/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- time
- [scikit-learn](https://pypi.org/project/scikit-learn/)
- [numpy](https://pypi.org/project/numpy/)

## Data
- [UCR2015](https://www.cs.ucr.edu/~eamonn/time_series_data/)

## Usage
```python
# Training the model without pre-training initialization
cd python train_original.py \
    --train_batch_size 128 \
    --val_batch_size   128 \
    --data_dir ../UCRArchive_2018/Plane \
    --gpu_nums 1 \
    --start_cuda_id 0\
    --max_epoch 500\
    --show_interval 50\
    --val_interval 1 \
    --out_dir Plane/Plane0 
# Training the pretext task A: Denoising 
cd denoise_pretrain && python denoise_longformer.py \
    --train_batch_size 64 \
    --val_batch_size   64 \
    --data_dir ../UCRArchive_2018/ArrowHead \
    --gpu_nums 1 \
    --start_cuda_id 1\
    --max_epoch 500\
    --show_interval 50\
    --val_interval 1 \
    --seq_len 128 \
    --out_dir ArrowHead_denoise_model_0_50 \
    --global_index_num 0 \
    --attention_window 50 \
    --beta 175\
# Initializing the model with the parameters trained by the pretext task A: Denoising.
cd denoise_pretrain && python test_fz_part.py\
    --train_batch_size 64 \
    --val_batch_size   64 \
    --data_dir ../UCRArchive_2018/UWaveGestureLibraryX \
    --gpu_nums 1 \
    --start_cuda_id 1\
    --max_epoch 500\
    --show_interval 50\
    --val_interval 1 \
    --seq_len 128 \
    --out_dir UWaveGestureLibraryX_denoise/UWaveGestureLibraryX_denoise0 \
    --global_index_num 0 \
    --attention_window 104 \
    --beta 60\
    --model_path ./Mallat_denoise_model/model_best_0.0015279224970274501.pth.tar
# Training the pretext task B: Similarity Discrimination Based on DTW.
cd dtw_pretrain && python dtw_longformer_train.py\
    --train_batch_size 16 \
    --val_batch_size   16 \
    --data_dir ../UCRArchive_2018/Mallat \
    --gpu_nums 1 \
    --start_cuda_id 0\
    --max_epoch 500\
    --show_interval 50\
    --val_interval 1 \
    --seq_len 128 \
    --out_dir Mallat_dtw1 \
    --global_index_num 0 \
    --attention_window 128 \
    --similar_path ../DTW_ED/similar_result
# Initializing the model with the parameters trained by the pretext taskB: Similarity Discrimination Based on DTW.
cd dtw_pretrain && python test_fz_part.py \
    --train_batch_size 128 \
    --val_batch_size   128 \
    --data_dir ../UCRArchive_2018/Adiac \
    --gpu_nums 1 \
    --start_cuda_id 0\
    --max_epoch 500\
    --show_interval 50\
    --val_interval 1 \
    --seq_len 128 \
    --out_dir Adiac_dtwn/Adiac_dtw0 \
    --global_index_num 0 \
    --attention_window 120 \
    --model -1\
    --model_path ./Mallat_dtw1/model_best_0.476027839978536.pth.tar
```
