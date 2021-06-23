# Video-Captioning-AV

This repository contains the code for the Final Project of the Natural Language Processing course at Tsinghua University, spring 2021.

In this project, we have developed a system that can generate a description of a short video clip using a sentence in English. To run our code, please follow these instructions:

## Environment setup

Make sure that you have Python 3 installed in your system. Also, Pytorch 1.5 or above needs to be installed. Check the [official installation guide](https://pytorch.org/get-started/locally/) to set it up according to your system requirements and CUDA version.

It is recommended to create a virtual environment to install the dependencies. Open a new terminal in the master directory, activate the virtual environment and install the dependencies from ``requirements.txt`` by executing this command:

```
$ (venv) pip install -r requirements.txt
```

## Data preparation

Download the MSVD and MSR-VTT datasets and place them in the `datasets/` folder. The code is expecting to have following structure: 

```
<dataset name>/
    audios/
        video1.wav
        ...
        videoN.wav
    features/
        audio/
        video/
    metadata/
        train.csv
        val.csv
        test.csv
    videos/
        video1.mp4
        ...
        videoN.mp4
```

Some folders inside `features/*` will be empty, and its contents will be generated after the *feature extraction* process.

> In case of the MSVD dataset, it is necessary to run the `download_youtube.py` script before to obtain the audio data.

## Feature extraction

After donwloading the data, we need to compute the features for each video and audio clip. 

```
(venv) python extract_features.py   --dataset <path_to_dataset_root>
                                    [--gpu <device_id>]
```

When the feature extraction finishes, the folders `features/audio` and `features/video` should contain 1 `.npy` file for each video in the dataset, with the same name as the original.

## Model training

To train a model, run the script `trainer.py` with the appropriate command line arguments, as follows:

```
(venv) trainer.py   --dataset 'MSVD' or 'MSR-VTT'
                    [--epochs <integer> (default, 50)]
                    [--batch_size <integer> (default, 128)]
                    [--lr <float> (default, 1e-4)]
                    [--gpu <device_id>] 
```

Modify the experiment configuration inside of the script and adapt it to your own needs. An array of experiments can also be used to run sequentially, one after the other, for conveninence.

## Evaluation
Clone evaluation codes from [python3 coco-evaluation repo](https://github.com/daqingliu/coco-caption) and copy them in the `pycocoevalcap/` folder.

**It is important to perform this step before training**, since the training script relies on the evaluation metrics for validation. 

When the training has finished, it is also possible to evaluate the results by loading an existing model checkpoint. To do that, execute the notebook in `nb/predict_captions.py`. The results will be saved to the folder `results/<dataset_name>` in `.csv` format with 1 file per model, containing the generated captions and ground truth captions. 

## Acknowledgements

* The code borrows the LSTM and reconstructor architectures from the [original RecNet implementation](https://github.com/hobincar/RecNet), which has been adapted to our multi-modal problem. 

* The VGGish model used for audio feature extraction is from [this PyTorch implementation](https://github.com/harritaylor/torchvggish).