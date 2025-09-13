from huggingface_hub import HfApi
api = HfApi()

# huggingface-cli repo create length_penalty_4b --type model

'''
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /home/user/checkpoints/length_penalty/global_step_1100/actor \
    --target_dir /home/user/hf_checkpoints/length_penalty
'''

api.upload_large_folder(
    folder_path="/home/user/hf_checkpoints/length_penalty",
    repo_id="vatsalb/length_penalty_4b",
    repo_type="model",
)