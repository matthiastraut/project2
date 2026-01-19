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

All model training takes place in notebook.ipynb. The model is then saved as model.onnx. (The accurace of the model isn't fantastic. One way to improve it is to use a pre-trained model but the focus in this project was on the model training process.)

## Deployment

The model is deployed using ONNX Runtime and FastAPI. The deployment script is model_deploy.py. To run the model locally, use:

```bash
uv sync --locked
python model_deploy.py
```

You can then access the backend here: http://0.0.0.0:9696/docs#/

The Dockerfile produces the corresponding image. We produced the image both locally and on GitHub Actions:

```bash
docker build -t aircraft-model .
docker run -p 9696:9696 aircraft-model
```

You can then access the backend here: http://0.0.0.0:9696/docs#/

## Server Deployment

One option to deploy the model is to use AWS Lambda. The deployment script is lambda_function.py but we chose to deploy instead on Render.com:

https://project2-n7kh.onrender.com/docs

To test the model, paste the following image link (or any other image link in jpg format):

https://cdn.plnspttrs.net/19481/g-civr-british-airways-boeing-747-436_PlanespottersNet_333483_9791a7bf0e_o.jpg