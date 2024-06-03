# QA Deployment

This document provides step-by-step instructions for deploying the QA deployment Docker container to Google Cloud.

## Prerequisites

- Install Docker
- Install the Google Cloud SDK and authenticate (`gcloud auth login`)
- Set up a Google Cloud project and ensure billing is enabled

## Deploying to Google Cloud

### 1. Build the Docker Image

Navigate to the `qa_deployment` directory:

```sh
cd finboost-ml/qa_deployment
```

Build the Docker image:

<!-- ```sh
docker build -t gcr.io/[PROJECT-ID]/qa_deployment .
``` -->

```sh
docker build -t qa_deployment .
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
docker tag qa_deployment:latest asia-southeast2-docker.pkg.dev/ents-h115/finboost-ml/qa_deployment:latest
```

Push the Docker image:

<!-- ```sh
docker push gcr.io/[PROJECT-ID]/qa_deployment
``` -->

```sh
docker push asia-southeast2-docker.pkg.dev/ents-h115/finboost-ml/qa_deployment:latest
```

### 3. Deploy to Google Cloud Run

Deploy the container image to Cloud Run:

```sh
gcloud run deploy qa-deployment --image asia-southeast2-docker.pkg.dev/ents-h115/finboost-ml/qa_deployment:latest --platform managed --region asia-southeast2 --allow-unauthenticated

```

### 4. Verify the Deployment

- Testing the Service with Postman
