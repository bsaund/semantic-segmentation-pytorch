# Overview
This repo is [forked](https://github.com/CSAILVision/semantic-segmentation-pytorch). The [original readme](original_readme/README.md) contains many details about the package.

This readme contains instructions for retraining a model based on your data

## YCB Video Dataset
I retrained on the [YCB Video dataset](https://github.com/yuxng/YCB_Video_toolbox).

1. download the [dataset](https://rse-lab.cs.washington.edu/projects/posecnn/) 
2. Move (or create a symlink to) `YCB_Video_Dataset` in the `data` folder
3. prepare the dataset and training/test file lists by running `prepare_ycb_video_dataset.py`
4. Make your config file in the `config` directory. Choose your model, dataset, hyperparams, etc.
5. `python train.py --cfg [your config file] --gpus [your gpu(s)]`

### Add your custom images
To tailor the segmenter to your custom environment, I suggest:

1. Take pictures of your scene WITHOUT any ycb objects
2. Overlay the YCB objects from the `data_syn` portion of the YCB_Video_Dataset. These are YCB objects on a transparent background.
3. Include these at a 50/50 ratio with the original YCB Video dataset.

Alternatively, you could create your own dataset by segmenting the YCB objects from a bunch of pictures you take. That seems like a lot of work though....
