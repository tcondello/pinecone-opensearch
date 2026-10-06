#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:=us-east-1}"
: "${EC2_INSTANCE_TYPE:=c7i.large}"
export AWS_REGION

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="$ROOT/.aws-ec2"
KEY="$DIR/key.pem"
STATE="$DIR/state"
REMOTE_DIR="pinecone-opensearch"
mkdir -p "$DIR"

get() { { grep "^$1=" "$STATE" 2>/dev/null || true; } | cut -d= -f2-; }
put() { touch "$STATE"; grep -v "^$1=" "$STATE" >"$STATE.tmp" || true; echo "$1=$2" >>"$STATE.tmp"; mv "$STATE.tmp" "$STATE"; }
ssh_opts() { echo "-i $KEY -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$DIR/known_hosts -o ConnectTimeout=10"; }
host() { echo "ec2-user@$(get ip)"; }

cmd_create() {
  local suffix name vpc sg ami my_ip iid ip
  suffix="$(get suffix)"; [[ -n "$suffix" ]] || { suffix="$(openssl rand -hex 3)"; put suffix "$suffix"; }
  name="pc-os-runner-$suffix"
  if [[ ! -f "$KEY" ]]; then
    aws ec2 create-key-pair --key-name "$name" --key-type ed25519 --query KeyMaterial --output text >"$KEY"
    chmod 600 "$KEY"
  fi
  vpc="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
  sg="$(get sg)"
  if [[ -z "$sg" ]]; then
    sg="$(aws ec2 create-security-group --group-name "$name" --description "pinecone-opensearch latency runner" --vpc-id "$vpc" --query GroupId --output text)"
    put sg "$sg"
  fi
  my_ip="$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')"
  aws ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp --port 22 --cidr "$my_ip/32" >/dev/null 2>&1 || true
  ami="$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 --query Parameter.Value --output text)"
  iid="$(aws ec2 run-instances --image-id "$ami" --instance-type "$EC2_INSTANCE_TYPE" --key-name "$name" \
    --security-group-ids "$sg" --associate-public-ip-address \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=20,VolumeType=gp3}' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$name},{Key=project,Value=pinecone-opensearch},{Key=owner,Value=$USER}]" \
    --query 'Instances[0].InstanceId' --output text)"
  put instance "$iid"
  echo "launched $iid ($EC2_INSTANCE_TYPE, $ami); waiting for running"
  aws ec2 wait instance-running --instance-ids "$iid"
  cmd_refresh_ip
  echo "waiting for ssh on $(get ip)"
  for i in $(seq 1 30); do
    ssh $(ssh_opts) "$(host)" true 2>/dev/null && break
    printf '  ssh attempt %d/30\n' "$i"; sleep 5
  done
  "$ROOT/scripts/aws/opensearch_domain.sh" allow-ip "$(get ip)"
}

cmd_refresh_ip() {
  put ip "$(aws ec2 describe-instances --instance-ids "$(get instance)" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
  echo "instance $(get instance) ip $(get ip)"
}

cmd_setup() {
  ssh $(ssh_opts) "$(host)" 'command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh'
  cmd_sync
  ssh $(ssh_opts) "$(host)" "cd $REMOTE_DIR && ~/.local/bin/uv sync --group dev && ~/.local/bin/uv run pytest -q | tail -1"
}

cmd_sync() {
  "$ROOT/scripts/aws/opensearch_domain.sh" endpoint | tail -1 >"$DIR/os_hosts.json"
  rsync -az --delete -e "ssh $(ssh_opts)" \
    --exclude .venv --exclude .git --exclude __pycache__ --exclude .aws-ec2 \
    --exclude 'live/data/corpus.emb.jsonl' --exclude 'live/out' \
    "$ROOT/" "$(host):$REMOTE_DIR/"
  scp $(ssh_opts) -q "$DIR/os_hosts.json" "$(host):$REMOTE_DIR/live/os_hosts.json"
  echo "synced to $(host):$REMOTE_DIR"
}

cmd_run() {
  ssh $(ssh_opts) "$(host)" "cd $REMOTE_DIR && export PATH=\$HOME/.local/bin:\$PATH && $*"
}

cmd_fetch() {
  mkdir -p "$ROOT/live/out/ec2"
  rsync -az -e "ssh $(ssh_opts)" "$(host):$REMOTE_DIR/live/out/" "$ROOT/live/out/ec2/"
  echo "fetched to live/out/ec2/"
}

cmd_stop() {
  aws ec2 stop-instances --instance-ids "$(get instance)" --query 'StoppingInstances[0].CurrentState.Name' --output text
  "$ROOT/scripts/aws/opensearch_domain.sh" revoke-ip "$(get ip)"
}

cmd_start() {
  aws ec2 start-instances --instance-ids "$(get instance)" >/dev/null
  aws ec2 wait instance-running --instance-ids "$(get instance)"
  cmd_refresh_ip
  "$ROOT/scripts/aws/opensearch_domain.sh" allow-ip "$(get ip)"
}

cmd_terminate() {
  local ip
  ip="$(get ip)"
  aws ec2 terminate-instances --instance-ids "$(get instance)" >/dev/null
  aws ec2 wait instance-terminated --instance-ids "$(get instance)"
  aws ec2 delete-security-group --group-id "$(get sg)"
  aws ec2 delete-key-pair --key-name "pc-os-runner-$(get suffix)"
  "$ROOT/scripts/aws/opensearch_domain.sh" revoke-ip "$ip"
  rm -rf "$DIR"
}

cmd="${1:-}"
shift || true
case "$cmd" in
  create) cmd_create ;;
  setup) cmd_setup ;;
  sync) cmd_sync ;;
  run) cmd_run "$@" ;;
  ssh) exec ssh $(ssh_opts) "$(host)" ;;
  fetch) cmd_fetch ;;
  stop) cmd_stop ;;
  start) cmd_start ;;
  terminate) cmd_terminate ;;
  *) echo "usage: $0 {create|setup|sync|run <cmd>|ssh|fetch|stop|start|terminate}" >&2; exit 2 ;;
esac
