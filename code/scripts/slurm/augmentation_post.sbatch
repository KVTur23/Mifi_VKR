#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=t2n1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --job-name=aug_post
#SBATCH --output=/mnt/pool/6/kvturanosov/VKR/code/logs/aug_post_%j.log
#SBATCH --error=/mnt/pool/6/kvturanosov/VKR/code/logs/aug_post_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/augmentation_post.sh"
