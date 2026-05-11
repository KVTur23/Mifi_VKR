#!/bin/bash
set -e

cd /mnt/pool/6/kvturanosov/VKR/worktrees/re-augmentation/code

SBATCH_ARGS=${SBATCH_ARGS:-}
GPU_PROFILE=${GPU_PROFILE:-A100_40}
TMP_GROUP=${TMP_GROUP:-iter4_cw}
FILTER=${FILTER:-}
BASE_RUN_DIR=/mnt/pool/6/kvturanosov/VKR/finetune_runs

submit_one () {
    local run_name="$1"
    local model_dir="$2"
    local run_time="$3"
    local job_name="$4"

    if [ -n "$FILTER" ] && [[ "$run_name" != *"$FILTER"* ]]; then
        return 0
    fi

    mkdir -p "$BASE_RUN_DIR/$run_name/logs"

    local pre
    pre=$(sbatch --parsable $SBATCH_ARGS \
        --job-name="${job_name}_pre" \
        --output="$BASE_RUN_DIR/$run_name/logs/pre_%j.log" \
        --error="$BASE_RUN_DIR/$run_name/logs/pre_%j.err" \
        --export=ALL,RUN_NAME="$run_name",MODEL_DIR="$model_dir",TMP_GROUP="$TMP_GROUP" \
        scripts/slurm/finetune_iter4_cw_pre.sbatch)
    echo "$run_name pre:  $pre"

    local run
    run=$(sbatch --parsable $SBATCH_ARGS \
        --dependency=afterok:$pre \
        --time="$run_time" \
        --job-name="$job_name" \
        --output="$BASE_RUN_DIR/$run_name/logs/run_%j.log" \
        --error="$BASE_RUN_DIR/$run_name/logs/run_%j.err" \
        --export=ALL,RUN_NAME="$run_name",TMP_GROUP="$TMP_GROUP",GPU_PROFILE="$GPU_PROFILE" \
        scripts/slurm/finetune_iter4_cw_run.sbatch)
    echo "$run_name run:  $run"

    local post
    post=$(sbatch --parsable $SBATCH_ARGS \
        --dependency=afterany:$run \
        --job-name="${job_name}_post" \
        --output="$BASE_RUN_DIR/$run_name/logs/post_%j.log" \
        --error="$BASE_RUN_DIR/$run_name/logs/post_%j.err" \
        --export=ALL,RUN_NAME="$run_name",TMP_GROUP="$TMP_GROUP" \
        scripts/slurm/finetune_iter4_cw_post.sbatch)
    echo "$run_name post: $post"
}

submit_one qwen25_32b_adalora_iter4_cw            models--Qwen--Qwen2.5-32B-Instruct                       12:00:00 q25_a4cw
submit_one qwen25_32b_qlora_iter4_cw              models--Qwen--Qwen2.5-32B-Instruct                       12:00:00 q25_q4cw

submit_one qwen3_14b_adalora_iter4_cw             models--Qwen--Qwen3-14B                                  10:00:00 q14_a4cw
submit_one qwen3_14b_lora_iter4_cw                models--Qwen--Qwen3-14B                                  10:00:00 q14_l4cw
submit_one qwen3_14b_qlora_iter4_cw               models--Qwen--Qwen3-14B                                  10:00:00 q14_q4cw
submit_one qwen3_14b_tinylora_iter4_cw            models--Qwen--Qwen3-14B                                  10:00:00 q14_t4cw

submit_one qwen3_32b_adalora_iter4_cw             models--Qwen--Qwen3-32B                                  12:00:00 q32_a4cw
submit_one qwen3_32b_qlora_iter4_cw               models--Qwen--Qwen3-32B                                  12:00:00 q32_q4cw

submit_one ruadapt_qwen3_32b_adalora_iter4_cw     models--RefalMachine--RuadaptQwen3-32B-Instruct          12:00:00 radp_a4cw
submit_one ruadapt_qwen3_32b_qlora_iter4_cw       models--RefalMachine--RuadaptQwen3-32B-Instruct          12:00:00 radp_q4cw

submit_one vikhr_nemo_12b_adalora_iter4_cw        models--Vikhrmodels--Vikhr-Nemo-12B-Instruct-R-21-09-24  10:00:00 vik_a4cw
submit_one vikhr_nemo_12b_lora_iter4_cw           models--Vikhrmodels--Vikhr-Nemo-12B-Instruct-R-21-09-24  10:00:00 vik_l4cw
submit_one vikhr_nemo_12b_qlora_iter4_cw          models--Vikhrmodels--Vikhr-Nemo-12B-Instruct-R-21-09-24  10:00:00 vik_q4cw
submit_one vikhr_nemo_12b_tinylora_iter4_cw       models--Vikhrmodels--Vikhr-Nemo-12B-Instruct-R-21-09-24  10:00:00 vik_t4cw

submit_one vistral_24b_adalora_iter4_cw           models--Vikhrmodels--Vistral-24B-Instruct                12:00:00 vist_a4cw
submit_one vistral_24b_qlora_iter4_cw             models--Vikhrmodels--Vistral-24B-Instruct                12:00:00 vist_q4cw

submit_one yandexgpt_5_lite_8b_adalora_iter4_cw   models--yandex--YandexGPT-5-Lite-8B-instruct             8:00:00 yan_a4cw
submit_one yandexgpt_5_lite_8b_lora_iter4_cw      models--yandex--YandexGPT-5-Lite-8B-instruct             8:00:00 yan_l4cw
submit_one yandexgpt_5_lite_8b_qlora_iter4_cw     models--yandex--YandexGPT-5-Lite-8B-instruct             8:00:00 yan_q4cw
submit_one yandexgpt_5_lite_8b_tinylora_iter4_cw  models--yandex--YandexGPT-5-Lite-8B-instruct             8:00:00 yan_t4cw

squeue -u "$(whoami)"
