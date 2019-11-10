### Total Denoising: *Unsupervised Learning of 3D Point Cloud Cleaning*
Created by <a href="https://www.uni-ulm.de/en/in/mi/institute/mi-mitarbeiter/pedro-hermosilla-casajus/" target="_blank">Pedro Hermosilla</a>, <a href="http://www.homepages.ucl.ac.uk/~ucactri/">Tobias Ritschel</a>, <a href="https://www.uni-ulm.de/in/mi/institut/mi-mitarbeiter/tr/" target="_blank">Timo Ropinski</a>.

![teaser](https://github.com//phermosilla/TotalDenoising/blob/master/teaser/Teaser.png)

### Citation
If you find this code useful please consider citing us:

        @article{hermosilla2019totaldenoise,
          title={Total Denoising: Unsupervised Learning of 3D Point Cloud Cleaning},
          author={Hermosilla, P. and Ritschel, and Ropinski, T.},
          journal={International Conference on Computer Vision 2019 (ICCV19)},
          year={2019}
        }

### Installation
First, install <a href="https://www.tensorflow.org/install/">TensorFlow</a>. The code presented here was developed using TensorFlow v1.13 GPU version, Python 3, and Ubuntu 16.04 TLS. All the operation were implemented on the GPU, no CPU implementation is provided. Therefore, a workstation with a state-of-the-art GPU is required.

#### Intalation of MCCNN library
Then, we need to download the MCCNN library into a folder named <a href="https://github.com/viscom-ulm/MCCNN">MCCNN</a> and follow the instructions provided in the readme file to compile the library.

#### Compiling tensorflow operations
In order to train the networks provided in this repository, first, we have to compile the new tensor operations. These operations are located on the folder `tf_ops`. To compile them we should first modify the paths on the file `compile.sh`. Then, we should execute the following commands:

    cd tf_ops
    sh compile.sh

### Datasets

#### Synthetic dataset

We will provide such dataset soon.

#### Rue Madame

You can download the dataset from the following <a href="http://www.cmm.mines-paristech.fr/~serna/rueMadameDataset.html">link</a>. Create a folder named `RueMadame`with the ply files on it and use the script `GenerateRueMadameDataSet.py` to subdivide the scan into chunks.

### Train

Use the script `Train.py` to train a model in the selected dataset. Use the command `Train.py --help` to look at the options provided by the script. The command used to train on the RueMadame dataset:

    python Train.py --dataset 3

### Test

Use the script `Test.py` to test a trained model. Use the command `Test.py --help` to look at the options provided by the script. The command to test a trained model on the RueMadame dataset:

    python3 Test.py --gaussFilter --dataset 3 --saveModels --noCompError

The command to test a trained model on one of the synthetic datasets:

    python3 Test.py --gaussFilter --dataset 0 --saveModels

### License
Our code is released under MIT License (see LICENSE file for details).