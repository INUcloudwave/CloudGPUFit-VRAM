#!/usr/bin/env bash
# Read-only validation of the user's selected AMI. No fallback or launch.
set -euo pipefail
region=${1:-us-east-1}
selected=ami-0bc11c94e8c4894e4
aws ec2 describe-images --region "$region" --image-ids "$selected" --query 'Images[].[ImageId,Name,State,Architecture,OwnerId]' --output table
state=$(aws ec2 describe-images --region "$region" --image-ids "$selected" --query 'Images[0].State' --output text)
architecture=$(aws ec2 describe-images --region "$region" --image-ids "$selected" --query 'Images[0].Architecture' --output text)
[[ "$state" == available && "$architecture" == x86_64 ]] || { printf 'Selected AMI unavailable or incompatible. Check region/ID; no automatic substitution.\n' >&2; exit 1; }
printf 'SELECTED_AMI=%s REGION=%s\n' "$selected" "$region"
