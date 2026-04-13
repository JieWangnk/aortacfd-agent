# AWS Deployment for aortacfd-agent

## Architecture

```
User (browser/CLI)
       |
       v
 t3.small EC2          <-- Always-on: Streamlit UI + API ($15/mo)
 (agent layer)               Runs intake/literature/config (LLM calls)
       |
       v  submit job
 AWS Batch (spot)       <-- On-demand: OpenFOAM CFD execution (~$0.10/hr)
 c5.2xlarge                  Scales to zero when idle
       |
       v  upload results
 S3 bucket              <-- STL inputs + CFD results (pennies/month)
```

Typical cost: ~$20-25/month for 20 CFD runs.

## Setup Steps

### 1. Push Docker image to ECR

```bash
# Create ECR repository (one-time)
aws ecr create-repository --repository-name aortacfd-agent --region eu-west-2

# Login, build, push
aws ecr get-login-password --region eu-west-2 | \
  docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.eu-west-2.amazonaws.com

docker build -t aortacfd-agent ..
docker tag aortacfd-agent:latest <ACCOUNT_ID>.dkr.ecr.eu-west-2.amazonaws.com/aortacfd-agent:latest
docker push <ACCOUNT_ID>.dkr.ecr.eu-west-2.amazonaws.com/aortacfd-agent:latest
```

### 2. Create S3 bucket for cases and results

```bash
aws s3 mb s3://aortacfd-runs --region eu-west-2
```

### 3. Deploy Batch infrastructure

```bash
# Register the job definition
aws batch register-job-definition --cli-input-json file://batch-job-definition.json

# Create compute environment (see batch-compute-env.json)
aws batch create-compute-environment --cli-input-json file://batch-compute-env.json

# Create job queue
aws batch create-job-queue \
  --job-queue-name aortacfd-queue \
  --state ENABLED \
  --priority 1 \
  --compute-environment-order order=1,computeEnvironment=aortacfd-spot-env
```

### 4. Submit a job

```bash
# Upload case data
aws s3 sync cases_input/BPM120/ s3://aortacfd-runs/input/BPM120/

# Submit
aws batch submit-job \
  --job-name "BPM120-standard" \
  --job-queue aortacfd-queue \
  --job-definition aortacfd-cfd-run \
  --container-overrides '{
    "environment": [
      {"name": "CASE_ID", "value": "BPM120"},
      {"name": "S3_BUCKET", "value": "aortacfd-runs"},
      {"name": "ANTHROPIC_API_KEY", "value": "sk-..."}
    ]
  }'

# Monitor
aws batch describe-jobs --jobs <JOB_ID> --query 'jobs[0].status'

# Download results
aws s3 sync s3://aortacfd-runs/output/BPM120/ ./output/BPM120/
```

## Cost Controls

- Batch uses **spot instances** by default (70-90% cheaper than on-demand)
- Set `maxvCpus: 16` in compute env to cap spending
- Add a CloudWatch billing alarm:
  ```bash
  aws cloudwatch put-metric-alarm \
    --alarm-name aortacfd-budget \
    --metric-name EstimatedCharges \
    --namespace AWS/Billing \
    --statistic Maximum \
    --period 21600 \
    --threshold 50 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 1 \
    --alarm-actions <SNS_TOPIC_ARN>
  ```
