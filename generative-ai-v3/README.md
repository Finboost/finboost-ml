# Deployment

This document provides step-by-step instructions for deploying the Generative AI deployment Docker container to Google Cloud.

## Prerequisites

- Install Docker
- Install the Google Cloud SDK and authenticate (`gcloud auth login`)
- Set up a Google Cloud project and ensure billing is enabled

## Deploying to Google Cloud

### 1. Deploy ke Google Cloud Run

Navigate to the `generative-ai-v3` directory:

```sh
cd finboost-ml/generative-ai-v3
```

Inisialisasi Google Cloud SDK

```sh
gcloud init
```

### 2. Aktifkan Cloud Run API

```sh
gcloud services enable run.googleapis.com

```

### 3 Setel Project ID dan Zone

```sh
gcloud config set project ents-h115
gcloud config set compute/zone asia-southeast2-a
gcloud config set compute/region asia-southeast2

```

### 4. Deploy to Google Cloud Run

Set Project

```sh
gcloud builds submit --config cloudbuild.yaml .
```

- Jika pake API_KEY

```sh
gcloud builds submit --config cloudbuild.yaml . --substitutions=_GROQ_API_KEY="your_actual_groq_api_key"

```

### 5. Verify the Deployment

- Auth

```sh
gcloud auth login
```

- Dapatkan Token Auth

```sh
for /f "tokens=*" %i in ('gcloud auth print-identity-token') do set TOKEN=%i
```

- Testing the Service with Postman
