# Deployment

This document provides step-by-step instructions for deploying the Generative AI deployment Docker container to Google Cloud.

## Prerequisites

- Install Docker
- Install the Google Cloud SDK and authenticate (`gcloud auth login`)
- Set up a Google Cloud project and ensure billing is enabled

## Deploying to Google Cloud

### 1. Build the Docker Image

Navigate to the `finboost-assistant` directory:

```sh
cd finboost-ml/finboost-assistant
```

Build the Docker image:

<!-- ```sh
docker build -t gcr.io/[PROJECT-ID]/finboost-assistant .
``` -->

```sh
docker build -t finboost-assistant .
```

### 2. Push the Docker Image to Google Container Registry

Authenticate with Google Cloud:

```sh
gcloud auth configure-docker
```

<!-- Create repo

```sh
gcloud artifacts repositories create REPOSITORY-ID --repository-format=docker --location=southeast-asia2

``` -->

Tag the image with a registry name

```sh
docker tag finboost-assistant:latest asia-southeast2-docker.pkg.dev/ents-h115/finboost-ml/finboost-assistant:latest
```

Push the Docker image:

<!-- ```sh
docker push gcr.io/[PROJECT-ID]/finboost-assistant
``` -->

```sh
docker push asia-southeast2-docker.pkg.dev/ents-h115/finboost-ml/finboost-assistant:latest
```

### 3. Deploy to Google Cloud Run

Set Project

```sh
gcloud config set project ents-h115
```

Deploy the container image to Cloud Run:

<!-- ```sh
gcloud run deploy finboost-ml --image asia-southeast2-docker.pkg.dev/ents-h115/finboost-ml/finboost-assistant:latest --platform managed --region asia-southeast2 --allow-unauthenticated

``` -->

```sh
gcloud run deploy finboost-assistant --image asia-southeast2-docker.pkg.dev/ents-h115/finboost-ml/finboost-assistant:latest --platform managed --region asia-southeast2 --allow-unauthenticated

```

### 4. Verify the Deployment

- Testing the Service with Postman
