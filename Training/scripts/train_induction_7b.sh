set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# sleep 14400
BASE_CONFIG="\
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=8192 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='meta-ability-alignment' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=120 \
    trainer.test_freq=100 \
    trainer.total_epochs=8"

# Define the ppl values for curriculum learning

PPL_VALUES="1 2"

# Initial model path
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
EXPERIMENT_NAME="Induction-7B"


for ppl in $PPL_VALUES; do
    echo "Starting training for level ${ppl}"

    TRAIN_FILE="../dataset/Induction/${ppl}/train.parquet"
    VAL_FILE="../dataset/Induction/${ppl}/test.parquet" # standard bench
    
    # Define experiment name for this stage
    CURRENT_EXPERIMENT_NAME="${EXPERIMENT_NAME}-${ppl}"


    # Construct the command
    COMMAND="python3 -m verl.trainer.main_ppo \
        ${BASE_CONFIG} \
        data.train_files=${TRAIN_FILE} \
        data.val_files=${VAL_FILE} \
        actor_rollout_ref.model.path='\"${MODEL_PATH}\"' \
        trainer.experiment_name=${CURRENT_EXPERIMENT_NAME} \
        $@"

    echo "Executing command: ${COMMAND}"
    eval ${COMMAND}

    LATEST_CHECKPOINT=$(ls -d checkpoints/meta-ability-alignment/${CURRENT_EXPERIMENT_NAME}/actor/global_step_* | sort -V | tail -n 1)
    MODEL_PATH="${LATEST_CHECKPOINT}"
    echo "Latest checkpoint: ${MODEL_PATH}"

    # Update model path to the checkpoint of the current stage
    # MODEL_PATH="Logic-RL/checkpoints/${CURRENT_EXPERIMENT_NAME}/actor" # checkpoint path here
done

echo "Curriculum learning finished!"
