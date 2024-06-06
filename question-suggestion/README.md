### Kategori pertanyaan

- Reksadana (pasar uang, pendapatan tetap, saham, dll)
- Obligasi
- Saham
- Emas
- Cryptocurrency
- Manajemen keungan pribadi (anggaran dan perencanaan, pengelolaan utang, pensiun)
- Asuransi
- Makro ekonomi (interest rate, gdp, dll)
- Pajak

# Deployment

This document provides step-by-step instructions for deploying the Question Suggestion deployment Docker container to Google Cloud.

## Prerequisites

- Install Docker
- Install the Google Cloud SDK and authenticate (`gcloud auth login`)
- Set up a Google Cloud project and ensure billing is enabled

## Deploying to Google Cloud

### 1. Build the Docker Image

Navigate to the `question-suggestion` directory:

```sh
cd finboost-ml/question-suggestion
```

Build the Docker image:

<!-- ```sh
docker build -t gcr.io/[PROJECT-ID]/question-suggestion .
``` -->

```sh
docker build -t question-suggestion .
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
docker tag question-suggestion:latest asia-southeast2-docker.pkg.dev/ents-h115/finboost-ml/question-suggestion:latest
```

Push the Docker image:

<!-- ```sh
docker push gcr.io/[PROJECT-ID]/question-suggestion
``` -->

```sh
docker push asia-southeast2-docker.pkg.dev/ents-h115/finboost-ml/question-suggestion:latest
```

### 3. Deploy to Google Cloud Run

Set Project

```sh
gcloud config set project ents-h115
```

Deploy the container image to Cloud Run:

```sh
gcloud run deploy finboost-ml --image asia-southeast2-docker.pkg.dev/ents-h115/finboost-ml/question-suggestion:latest --platform managed --region asia-southeast2 --allow-unauthenticated

```

### 4. Verify the Deployment

- Testing the Service with Postman
