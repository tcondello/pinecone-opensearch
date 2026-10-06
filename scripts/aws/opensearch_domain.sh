#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:=us-east-1}"
: "${OS_ENGINE_VERSION:=OpenSearch_2.19}"
: "${OS_INSTANCE_TYPE:=t3.medium.search}"
: "${OS_VOLUME_GB:=20}"
export AWS_REGION

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STATE_FILE="$ROOT/.aws-domain"
IPS_FILE="$ROOT/.aws-domain-ips"

domain_name() {
  if [[ -f "$STATE_FILE" ]]; then
    cat "$STATE_FILE"
  else
    local name="pc-os-diff-$(LC_ALL=C tr -dc 'a-z0-9' </dev/urandom | head -c 6)"
    echo "$name" >"$STATE_FILE"
    echo "$name"
  fi
}

current_ip() {
  curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]'
}

allowed_ips_json() {
  sort -u "$IPS_FILE" | awk 'NF {printf "%s\"%s/32\"", (n++ ? "," : ""), $1}'
}

access_policy() {
  local name="$1" account
  account="$(aws sts get-caller-identity --query Account --output text)"
  echo "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"AWS\":\"*\"},\"Action\":\"es:ESHttp*\",\"Resource\":\"arn:aws:es:${AWS_REGION}:${account}:domain/${name}/*\",\"Condition\":{\"IpAddress\":{\"aws:SourceIp\":[$(allowed_ips_json)]}}}]}"
}

push_policy() {
  local name
  name="$(domain_name)"
  echo "Allowing $(sort -u "$IPS_FILE" | paste -sd, -) on ${name}"
  aws opensearch update-domain-config --domain-name "$name" \
    --access-policies "$(access_policy "$name")" \
    --query 'DomainConfig.AccessPolicies.Status.State' --output text
}

cmd_create() {
  local name
  name="$(domain_name)"
  current_ip >>"$IPS_FILE"
  echo "Creating domain ${name} (${OS_ENGINE_VERSION}, ${OS_INSTANCE_TYPE})"
  aws opensearch create-domain \
    --domain-name "$name" \
    --engine-version "$OS_ENGINE_VERSION" \
    --cluster-config "InstanceType=${OS_INSTANCE_TYPE},InstanceCount=1,ZoneAwarenessEnabled=false" \
    --ebs-options "EBSEnabled=true,VolumeType=gp3,VolumeSize=${OS_VOLUME_GB}" \
    --node-to-node-encryption-options Enabled=true \
    --encryption-at-rest-options Enabled=true \
    --domain-endpoint-options EnforceHTTPS=true \
    --access-policies "$(access_policy "$name")" \
    --tag-list Key=project,Value=pinecone-opensearch Key=owner,Value="${USER}" \
    --query 'DomainStatus.ARN' --output text
}

cmd_wait() {
  local name start status endpoint
  name="$(domain_name)"
  start=$(date +%s)
  while true; do
    read -r status endpoint < <(aws opensearch describe-domain --domain-name "$name" \
      --query '[DomainStatus.Processing, DomainStatus.Endpoint]' --output text)
    printf '[%4ds] %s processing=%s endpoint=%s\n' $(( $(date +%s) - start )) "$name" "$status" "$endpoint"
    if [[ "$status" == "False" && "$endpoint" != "None" ]]; then
      break
    fi
    sleep 30
  done
}

cmd_endpoint() {
  local endpoint
  endpoint="$(aws opensearch describe-domain --domain-name "$(domain_name)" \
    --query 'DomainStatus.Endpoint' --output text)"
  echo "https://${endpoint}"
  echo "[{\"host\":\"${endpoint}\",\"port\":443,\"use_ssl\":true}]"
}

cmd_allow_ip() {
  if [[ $# -eq 0 ]]; then
    current_ip >>"$IPS_FILE"
  else
    printf '%s\n' "$@" >>"$IPS_FILE"
  fi
  push_policy
}

cmd_revoke_ip() {
  local tmp
  tmp="$(mktemp)"
  grep -vxF -f <(printf '%s\n' "$@") "$IPS_FILE" >"$tmp" || true
  mv "$tmp" "$IPS_FILE"
  push_policy
}

cmd_delete() {
  local name
  name="$(domain_name)"
  aws opensearch delete-domain --domain-name "$name" --query 'DomainStatus.DomainName' --output text
  rm -f "$STATE_FILE" "$IPS_FILE"
}

cmd="${1:-}"
shift || true
case "$cmd" in
  create) cmd_create ;;
  wait) cmd_wait ;;
  endpoint) cmd_endpoint ;;
  allow-ip) cmd_allow_ip "$@" ;;
  revoke-ip) cmd_revoke_ip "$@" ;;
  delete) cmd_delete ;;
  *) echo "usage: $0 {create|wait|endpoint|allow-ip [ip...]|revoke-ip ip...|delete}" >&2; exit 2 ;;
esac
