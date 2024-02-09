# Understanding Model Competency

## 0) Set Up Codebase

### 0a. Clone this repo

Clone this repository:
```
git clone https://github.com/sarapohland/explainable-competency.git
```

### 0b. Set up the perception package

In the main folder (explainable-competency), run the following command:
```
pip install -e .
```

## 1) Set Up Datasets

### 1a. Download the dataset files

To replicate the results presented in [Understanding the Dependence of Perception Model Competency on Regions in an Image](https://www.overleaf.com/read/nxckwwgpkkkd#472312), download the Lunar, Speed, and Pavilion dataset files from the 'data' folder available [here](https://drive.google.com/drive/folders/1O9PvEPU7MHapiWA0Tk1_2SstNQyYhAuq?usp=share_link). Create a folder called 'data' in the  main directory (explainable-competency). For each dataset, create a subfolder in 'data' with the same name as the Drive folder from which you retrieved the file and place the dataset files in their corresponding subfolders. If you want to use these datasets only, you can skip to step 2. If you want to create additional datasets, proceed through the rest of the substeps in this section.

### 1b. Set up directory structure

By default, datasets are assumed to be saved in the following structure:
|-- data  
&emsp;|-- dataset1  
&emsp;&emsp;|-- dataset.p  
&emsp;&emsp;|-- images  
&emsp;&emsp;&emsp;|-- ID  
&emsp;&emsp;&emsp;&emsp;|-- class1  
&emsp;&emsp;&emsp;&emsp;|-- class2  
&emsp;&emsp;&emsp;|-- OOD  
&emsp;&emsp;&emsp;&emsp;|-- class1  
&emsp;&emsp;&emsp;&emsp;|-- class2  
&emsp;&emsp;&emsp;|-- unsorted  
&emsp;|-- dataset2  
&emsp;&emsp;|-- dataset.p  
&emsp;&emsp;|-- images   
&emsp;&emsp;&emsp;|-- ID  
&emsp;&emsp;&emsp;&emsp;|-- class1  
&emsp;&emsp;&emsp;&emsp;|-- class2  
&emsp;&emsp;&emsp;|-- OOD  
&emsp;&emsp;&emsp;&emsp;|-- class1  
&emsp;&emsp;&emsp;&emsp;|-- class2  
&emsp;&emsp;&emsp;|-- unsorted   

The unsorted folder should contain in-distribution training images that have not been labeled, while the ID folder contains all labeled in-distribution images organized by their class labels. If you already have a labeled dataset, you can organize them in the ID folder and skip to step 1d. If you only have unlabeled data, you can place it all in the unsorted folder and proceed to step 1c. The OOD folder should contain all out-of-distribution images. If this data is labeled, it can be orgnized into its class labels. If it is unlabeled, you can place it all into the same subfolder within the OOD folder. A dataset that has already been set up (following step 1d) will be saved in a pickle file called dataset.p in the main dataset folder.

### 1c. Cluster unlabeled data

If you have labeled data, skip to the next step. If you have unlabeled in-distribution data saved in the unsorted directory (from 1b), you can cluster these images using the create_dataset script:

```
python perception/datasets/create_dataset.py <path_to_dataset> --cluster_data
```

This command will cluster the unsorted images and save them in subfolders within the ID folder.

### 1d. Save custom dataset

Once you have existing classes of in-distribution data, you can save a dataset of training, test, and ood data using the create_dataset script:

```
python perception/datasets/create_dataset.py <path_to_dataset> --save_data
```

Note that this step can be combined with the previous one. By separating these two steps, you can validate the generated clusters before saving your dataset. You can also use to height and width flags to resize your images if desired. This script will save a pickle file called dataset.p in your dataset directory.

### 1e. Update dataloader setup script

Use the existing cases in the setup_dataloader script to enable the use of your custom dataset. You will need to add a section to the get_class_names, get_num_classes, and the setup_loader functions.

## 2) Generate Classification Model

### 2a. Download the configuration files

To replicate the results presented in [Understanding the Dependence of Perception Model Competency on Regions in an Image](https://www.overleaf.com/read/nxckwwgpkkkd#472312), download the models folder from [here](https://drive.google.com/drive/folders/1O9PvEPU7MHapiWA0Tk1_2SstNQyYhAuq?usp=share_link) and place it in the main directory (explainable-competency). This folder contains the model architectures and training parameters used to train the classification models used in this paper. If you want to modify the configurations to train new models, go through steps 2b and 2c. Otherwise, skip to step 2d. 

### 2b. Define the model architecture

Create a JSON file defining your model architecture using the example given in perception/network/layers.json. Currently, you can define simple model architectures composed of convolutional, pooling, and fully-connected (linear) layers with linear, relu, hyperbolic tangent, sigmoid, and softmax activation functions. You can also add a flattening layer in between other layers. For convolutional layers, you must specify the number of input and output channels and the kernel size. For pooling layers, you must specify the pooling function (max or average) and the kernel size. Finally, for fully-connected layers, you must specify the number of input and output nodes.

### 2c. Define the training parameters

Create a config file defining your training parameters using the example given in perception/network/train.config. You must specify the optimizer (sgd or adam), as well as the relevant optimizer parameters. Here you should also specify the desired loss function, number of epochs, and training/test batch sizes.

### 2d. Train the classification model

You can train your model using the train script in the network folder:

```
python perception/network/train.py --train_data lunar --output_dir models/lunar/classify/ --train_config models/lunar/classify/train.config --network_file models/lunar/classify/layers.json
```

The argument train_data is used to indicate which dataset should be used to train your classification model. Currently, only the emnist dataset is implemented, as well as a few custom datasets (lunar, speed, pavilion) that were used for evaluation in the paper. The argument output_dir is used to define where your trained model will be saved. (This should be models/<train_data>/classify if you downloaded the model folder in 2a.) The arguments network_file and train_config are used to specify the location of your model architecture json file (downloaded in step 2a or created in 2b) and training parameter config file (downloaded in step 2a or created in 2c). If you are using the downloaded models, then when prompted to overwrite the existing folder, you can type 'n'.

### 2e. Evaluate the classification model

You can evaluate your model using the test script in the network folder:

```
python perception/network/test.py --test_data lunar --model_dir models/lunar/classify/
```

The argument test_data is used to indicate which dataset should be used to evaluate your classification model. Currently, only the emnist dataset is implemented, as well as a few custom datasets (lunar, speed, and pavilion). The argument model_dir is used to specify where your trained classification model was saved. This should be the same location defined as the output_dir in step 2d.

## 3) Generate Competency Estimator

### 3a. Train the competency estimator

You can train a competency estimator for your classification model using the train script in the competency folder:

```
python perception/competency/train.py --train_data lunar --model_dir models/lunar/classify/
```

The argument train_data is used to indicate which dataset should be used to train your classification model. Currently, only the emnist dataset is implemented, as well as a few custom datasets (lunar, speed, and pavilion). The argument model_dir is used to specify where your trained classification model was saved and where your competency estimator will be saved. This should be the same location defined as the output_dir in step 2d.

### 3b. Evaluate the competency estimator

You can evaluate your competency estimator using the test script in the competency folder:

```
python perception/competency/test.py --test_data lunar --model_dir models/lunar/classify/
```

The argument test_data is used to indicate which dataset should be used to evaluate your classification model. Currently, only the emnist dataset is implemented, as well as a few custom datasets (lunar, speed, and pavilion). The argument model_dir is used to specify where your trained classification model and competency estimator were saved. This should be the same location defined as the model_dir in step 3a. This script will generate plots of various competency scores, which will be saved in a folder called results/<test_data>/competency. You can set the threshold for determining in-distribution vs. out-of-distribution data using the thresh parameter.

## 4) Tune Segmentation Parameters

Most of the approaches for understanding the regions contributing to low model competency rely on a segmentation algorithm. If you want to use the default segmentation parameters to replicate the results presented in [Understanding the Dependence of Perception Model Competency on Regions in an Image](https://www.overleaf.com/read/nxckwwgpkkkd#472312) for the datasets downloaded in step 1a, you can skip this step. If you created new datasets and would like to tune the hyperparameters for the segmentation algorithm, you can run the segment script:

```
python perception/regions/segment.py --img_folder data/Lunar/images/ID/ --output_dir data/Lunar/segmented/ --sigma 0.5 --scale 1000 --min_size 80
```

The img_folder specifies the location of the image files, which should either be the ID folder or OOD folder for a given dataset with the structure specified in 1b. The segmented images will be saved to the folder specified by ouput_dir. You can test different hyperparameters by changing the values of sigma, scale, and min_size. Optionally, you can also use to height and width flags to resize the image before segmenting it.

## 5) Generate Image Inpainting Model

### 5a. Download the configuration files

To replicate the results presented in [Understanding the Dependence of Perception Model Competency on Regions in an Image](https://www.overleaf.com/read/nxckwwgpkkkd#472312), download the models folder from [here](https://drive.google.com/drive/folders/1O9PvEPU7MHapiWA0Tk1_2SstNQyYhAuq?usp=share_link) and place it in the main directory (if you have not done so already). This folder contains the autoencoder architectures and training parameters used to train the image reconstruction and inpainting models used in this paper. If you want to modify the configurations to train new models, go through steps 5b and 5c. Otherwise, skip to step 5d. 

### 5b. Define the model architecture

Create two JSON files for the encoder and decoder portions of the autoencoder network using the examples given in perception/inpainting/encoder.json and perception/inpainting/decoder.json.

### 5c. Define the training parameters

Create a config file defining your training parameters using the example given in perception/inpainting/train.config. You must specify the optimizer (sgd or adam), as well as the relevant optimizer parameters. Here you should also specify the desired loss function, number of epochs, and training/test batch sizes.

### 5d. Construct a dataset for pretraining

You can pretrain the image inpainting model with the original (non-segmented) data. If your original images are not the correct size to be used as input to the autoencoder model, you must first resize the original dataset using the custom_dataset script:

```
python perception/datasets/custom_dataset.py <path_to_dataset> --height 120 --width 160
```

The path_to_dataset should be the location of a pickle file downloaded or generated in step 1 (e.g., data/Lunar/). If you are using the default autoencoder model architectures (downloaded in 5a), then the correct size of the input images is specified in the config file corresponding to each dataset under perception/configs/. This script will save a pickle file called original-dataset.p, which is very similar to the input dataset.p file, but these images will be of the desired size.

### 5e. Pretrain model with original dataset

It is recommended that you pretrain the image inpainting model with the original (non-segmented) data. To do so, you can use the train script in the inpainting folder:

```
python perception/inpainting/train.py <training_data> --output_dir models/lunar/reconstruct/ --train_config models/lunar/reconstruct/train.config --encoder_file models/lunar/reconstruct/encoder.json --decoder_file models/lunar/reconstruct/decoder.json --pretrain
```

The training_data should either be the pickle file generated in step 5d (e.g., data/Lunar/original-dataset.p) or the original one from step 1 if you did not need to resize the images (e.g., data/Lunar/dataset.p). The train_config contains the training parameters chosen in step 5a/c, and the encoder_file and decoder_file define the model architecture chosen in 5a/b. The trained model will be saved in the folder specified by output_dir. If you are using the downloaded models, then when prompted to overwrite the existing folder, you can type 'n'.

### 5f. Construct the segmented dataset

Before training the image inpainting model, you must first generate the segmented dataset using the custom_datatset script with the segment flag:

```
python perception/datasets/custom_dataset.py <path_to_dataset> --segment --sigma 0.5 --scale 750 --min_size 80 --height 120 --width 160
```

The path_to_dataset should be the location of the pickle file downloaded or generated in step 1 (e.g., data/Lunar/). You should use the best hyperparamters for sigma, k, and min_size that you found in step 4 (or the default ones provided in the config file corresponding to this dataset under perception/configs/). Optionally, you can also use to height and width flags to resize the image before segmenting it. If you are using the default autoencoder model architectures (downloaded in 5a), then the correct size of the input images (the height and width) is specified in the config file corresponding to each dataset under perception/configs/.

### 5g. Train initialized model with segmented dataset

You can train the image inpainting model with the segmented dataset generated in the previous step using the train script in the inpainting folder:

```
python perception/inpainting/train.py <training_data> --output_dir models/lunar/inpaint/ --train_config models/lunar/inpaint/train.config --encoder_file models/lunar/inpaint/encoder.json --decoder_file models/lunar/inpaint/decoder.json --init_model models/lunar/reconstruct/model.pth
```

The training_data should be the pickle file generated in step 5f (e.g., data/Lunar/segmented-dataset.p), and the init_model is the pretrained model from step 5d (if you completed this step). The train_config, encoder_file, and decoder_file are the same configuration files defined in step 5a-c. The trained model will be saved in the folder specified by output_dir. If you are using the downloaded models, then when prompted to overwrite the existing folder, you can type 'n'.

### 5h. Evaluate the image inpainting model

You can evaluate the image inpainting model generated in the previous step using the test script in the inpainting folder:

```
python perception/inpainting/test.py <test_data> --model_dir models/lunar/inpaint/
```

The test_data should be the pickle file generated in step 5e (and used for training), and the model_dir should be the output_dir from the previous step. If you want to evaluate the pretrained model before training with the segmented dataset, you can use the pretrain flag and replace test_data with the pickle file from either step 1 or 5d. This script will save figures of the original, masked, and reconstructed images to a folder called results in the directory specified by model_dir.

## 6) Understand Regions Contributing to Low Model Competency

You can specify the parameters for all of the regional competency understanding methods in a configuration file like the example given in perception/configs/lunar.config. To tune these parameters, you can use the scripts described in the following subsections. If you simply want to replicate the results from [Understanding the Dependence of Perception Model Competency on Regions in an Image](https://www.overleaf.com/read/nxckwwgpkkkd#472312), you can skip to step 7.

### 6a. Partitioning and cropping method

To test the partitioning and cropping method, use the cropping script in the regions folder:

```
python perception/regions/cropping.py --test_data lunar --model_dir models/lunar/classify/ --config_file perception/configs/lunar.config --output_dir results/lunar/crop/
```

This command will save regional competency images for the partitioning and cropping method to the folder specified by output_dir. Note that you can optionally use the debug flag to visualize the cropped images for each sample in the test set.

### 6b. Segmenting and masking method

To test the segmenting and masking method, use the masking script in the regions folder:

```
python perception/regions/masking.py --test_data lunar --model_dir models/lunar/classify/ --config_file perception/configs/lunar.config --output_dir results/lunar/mask/
```

This command will save regional competency images for the segmenting and masking method to the folder specified by output_dir. Note that you can optionally use the debug flag to visualize the masked images for each sample in the test set.

### 6c. Pixel perturbation method

To test the pixel perturbation method, use the perturbation script in the regions folder:

```
python perception/regions/perturbation.py --test_data lunar --model_dir models/lunar/classify/ --config_file perception/configs/lunar.config --output_dir results/lunar/perturb/
```

This command will save regional competency images for the pixel perturbation method to the folder specified by output_dir. Note that you can optionally use the debug flag to visualize the perturbed images for each sample in the test set.

### 6d. Pixel gradient method

To test the pixel gradient method, use the gradients script in the regions folder:

```
python perception/regions/gradients.py --test_data lunar --model_dir models/lunar/classify/ --config_file perception/configs/lunar.config --output_dir results/lunar/grads/
```

This command will save regional competency images for the pixel gradient method to the folder specified by output_dir. Note that you can optionally use the debug flag to visualize the competency gradients without averaging over image segments for each sample in the test set.

### 6e. Reconstruction loss method

To test the reconstruction loss method, use the reconstruction script in the regions folder:

```
python perception/regions/reconstruction.py --test_data lunar --model_dir models/lunar/classify/ --autoencoder_dir models/lunar/inpaint/ --config_file perception/configs/lunar.config --output_dir results/lunar/reconstr/
```

This command will save regional competency images for the reconstruction method to the folder specified by output_dir. Note that you can optionally use the debug flag to visualize the original image, the masked image used as input to the autoencoder, and the reconstructed image for each sample in the test set.

## 7) Compare Regional Competency Approaches

### 7a. Visually compare different approaches

To visually compare each of the five approaches contributing to low model competency, you can use the visualize script in the evaluation folder:

```
python perception/evaluation/visualize.py --test_data lunar --model_dir models/lunar/classify/ --autoencoder_dir models/lunar/inpaint/ --config_file perception/configs/lunar.config --output_dir results/lunar/figures/
```

This will save figures comparing the various approaches in the folder specified by output_dir. Note that you can optionally use the threshold argument so that pixels below the thresholds specified in the config file will be set to zero and pixels above will be set to one.

### 7b. Compare the computation times of approaches

You can compare the computation times for each of the five approaches using the computation script in the evaluation folder:

```
python perception/evaluation/computation.py --test_data lunar --model_dir models/lunar/classify/ --autoencoder_dir models/lunar/inpaint/ --config_file perception/configs/lunar.config
```

This will save a boxplot comparing the computation times of the approaches to a figure called times.png to the folder results/<test_data>. It will also print the average computation time for each method.

### 7c. Generate true labels of segemented regions

If you downloaded the Lunar, Speed, and/or Pavilion dataset files from the 'data' [folder](https://drive.google.com/drive/folders/1O9PvEPU7MHapiWA0Tk1_2SstNQyYhAuq?usp=share_link) in step 1a, you can also download the segmentation label files for these datasets from this same folder. These segmentation files should be placed in their corresponding subfolders (in the same way that they appear in the Drive data folder).

If you generated additional datasets in step 1, you need to manually generate the true unfamiliarity labels for each of the segmented regions in the test set. To do so, you can use the create_labels script in the evaluation folder:
```
python perception/evaluation/create_labels.py --test_data lunar --model_dir models/lunar/classify/ --config_file perception/configs/lunar.config
```

Each image in the test set will be segmented, and you will be shown each segmented region with the prompt: "Does this segment contain a structure not present in the training set?" Answering yes (y) will indicate that this region is unfamiliar to the model, while answering no (n) will indicate that it is familiar. After labeling the image, you will be shown the labeled image with the prompt: "Does this image need to be corrected?" Answering no (n) will save your labels and move on to the next image in the test set. If you answer yes (y), you will go through the labeling process again for this same test image. These responses will be saved to a pickle file called segmentation.p in a folder called data/<test_data>. Note that this file will be updated after you complete each image, so you can label portions of the test set at a time and resume where you left off. You can also review all of your labeled images using the test flag and indicate where you wish to begin reviewing images using the start_idx argument. While reviewing images, you will be asked whether each labeled image needs to be corrected.

### 7d. Compare the accuracy of approaches

You can compare the unfamiliarity prediction accuracy for each of the five approaches using the accuracy script in the evaluation folder:

```
python perception/evaluation/accuracy.py --test_data lunar --model_dir models/lunar/classify/ --autoencoder_dir models/lunar/inpaint/ --config_file perception/configs/lunar.config
```

This script will print various measures of accuracy (overall, TPR, TNR, PPV, and NPV) for each of the five approaches. It will also generate a boxplot comparing the overall accuracy of the five approaches, which will be saved to a figure called accuracy.png to the folder results/<test_data>. You can optionally use the debug flag to visualize the true unfamiliarity labels, along with the predicted labels for each of the five approaches. 

### 7e. Evaluate the accuracy of ensembling

You can compare the unfamiliarity prediction accuracy for ensembles of the main four approaches (masking, perturbation, gradients, and reconstruction) using the ensembling script in the evaluation folder:

```
python perception/evaluation/ensembling.py --test_data lunar --model_dir models/lunar/classify/ --autoencoder_dir models/lunar/inpaint/ --config_file perception/configs/lunar.config --thresh 0.9
```

This script will print various measures of accuracy and generate a boxplot comparing the overall accuracy of the ensembles, which will be saved to a figure called ensemble.png to the folder results/<test_data>. You can optionally use the debug flag to visualize the true unfamiliarity labels, along with the predicted labels for each of the ensembles. You can also change the threshold used on the z-score to determine whether a segment is familiar or unfamiliar to the perception model. 
