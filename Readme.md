# Delpoyment of a Pytorch Model using Django API

* Training a pytorch model on mnist dataset to predict the digit on image and predict the sum and differnce with a random number inputed to network with the image
* Deploying the trained model usnig Django API

## Requirements 

**Using pip**

```
pip3 install torch torchvision
python -m pip install Django
pip install tqdm
```

**Using conda (recommanded)**

```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install -c anaconda django
conda install -c conda-forge tqdm
```

## How to start

```
git clone https://github.com/anantgupta129/MNIST-Adder-Diff-With-Django-API.git
cd MNIST-Adder-Diff-With-Django-API
python train.py
python manage.py runserver
```

The test dataset images for MNIST are stored in [data directory](data/MNISTtestSample)


![](data/home.png)
Click on *try now*
![](data/input.png)
![](data/outscreen.png)

