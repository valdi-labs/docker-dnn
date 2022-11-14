# VALDI Sample Implementations

VALDI requires any programs, applications, or tasks to be bundled as Docker images. The goal of this repository is to 
provide code examples with Dockerfiles for running various use cases on the VALDI network.

**Note**: The first example is training a DNN on the MNIST dataset using Tensorflow. Eventually we should create 
subdirectories for each example use case (e.g., training a CNN, hosting a website, serving a model via API).

## Usage

The code trains a DNN on the MNIST dataset via the TensorFlow Datasets library. During training, it logs progress to a 
CloudWatch Log Stream. Upon completion, it uploads the final model to a permissioned S3 bucket.

### Running Locally

Install dependencies:
```angular2html
pip install -r requirements.txt
```

**Note**: To run locally on a Mac with Apple Silicon, replace `tensorflow==2.10.0` in the requirements.txt file with 
`tensorflow-macos==2.10.0`. 

Update configuration: 

Duplicate `config-example.yaml` as `config.yaml` and fill in the fields with actual values. The MODEL.NAME can be 
whatever you want; the rest is specific to your AWS account.

Run the code:
```angular2html
python main.py
```

The DNN training is hard coded to 100 epochs, but the training still finishes within 60-90 seconds.

### Using Docker

With Docker installed locally, you can use the standard Docker build command to create the image against the included 
`Dockerfile`:

```angular2html
docker build . -t [repo-name]:[tag]
```

The resulting image is over 1.5 GB. I tried experimenting with multi-stage builds (see `Dockerfile-multistage`) to 
reduce the size a bit, but to no avail (yet).

Run a container from the image:

```angular2html
docker run -d [repo-name]:[tag]
```