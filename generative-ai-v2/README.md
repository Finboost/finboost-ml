# Deployment

This document provides step-by-step instructions for deploying the Generative AI deployment Docker container to Google Cloud.

## Prerequisites

- Install Docker
- Install the Google Cloud SDK and authenticate (`gcloud auth login`)
- Set up a Google Cloud project and ensure billing is enabled

## Deploying to Google Cloud

### 1. Build the Docker Image

Navigate to the `generative-ai-v2` directory:

```sh
cd finboost-ml/generative-ai-v2
```

Build the Docker image:

<!-- ```sh
docker build -t gcr.io/[PROJECT-ID]/generative-ai-v2 .
``` -->

```sh
docker build -t generative-ai-v2 .
```

Run the Docker container locally to test (optional)

```sh
docker run -p 8080:8080 generative-ai-v2
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
docker tag generative-ai-v2:latest asia-southeast2-docker.pkg.dev/ents-h115/finboost-ml/generative-ai-v2:latest
```

Push the Docker image:

<!-- ```sh
docker push gcr.io/[PROJECT-ID]/generative-ai-v2
``` -->

```sh
docker push asia-southeast2-docker.pkg.dev/ents-h115/finboost-ml/generative-ai-v2:latest
```

### 3. Deploy to Google Cloud Run

Set Project

```sh
gcloud config set project ents-h115
```

Deploy the container image to Cloud Run:

<!-- ```sh
gcloud run deploy finboost-ml --image asia-southeast2-docker.pkg.dev/ents-h115/finboost-ml/generative-ai-v2:latest --platform managed --region asia-southeast2 --allow-unauthenticated

``` -->

```sh
gcloud run deploy generative-ai-v2 --image asia-southeast2-docker.pkg.dev/ents-h115/finboost-ml/generative-ai-v2:latest --platform managed --region asia-southeast2 --allow-unauthenticated

```

### 4. Verify the Deployment

- Testing the Service with Postman
