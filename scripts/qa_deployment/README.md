# Build and Deploy the Docker Container

- Install and Initialize Google Cloud SDK:
  curl https://sdk.cloud.google.com | bash
  exec -l $SHELL
  gcloud init

- Build the Docker Image:
  gcloud builds submit --tag gcr.io/PROJECT-ID/your-app
  Replace PROJECT-ID with your Google Cloud project ID.

- Deploy to Cloud Run:
  gcloud run deploy --image gcr.io/PROJECT-ID/your-app --platform managed
