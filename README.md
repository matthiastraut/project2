# Project

This project trains a neural network to identify airplanes, based on the FGVC Aircraft Benchmark, and subsequently deploys the model.

## Dataset

The dataset used for training the model is the FGVC Aircraft Benchmark, which can be found [here](https://github.com/cvml/fgvc-aircraft-2013.git).

For a more detailed description of the dataset, please see [here](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/).

The dataset can be downloaded via:

```bash

!wget https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
!untar fgvc-aircraft-2013b.tar.gz

```

## Training

All model training takes place in notebook.ipynb. The model is then saved as model.onnx.

## Deployment

The model is deployed using ONNX Runtime and FastAPI. The deployment script is model_deploy.py.

## Lambda Deployment

One option to deploy the model is to use AWS Lambda. The deployment script is lambda_function.py. The Dockerfile produces the corresponding image.