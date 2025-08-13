from huggingface_hub import HfApi
api = HfApi()

# huggingface-cli repo create your-repo-name --type model

api.upload_large_folder(
    folder_path="/home/user/checkpoints/weak_model_interp/global_step_400",
    repo_id="vatsalb/grpo_weak_model_interp",
    repo_type="model",
)