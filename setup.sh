conda create -n verl python==3.10 -y
conda activate verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

pip install opentelemetry-api==1.26.0
pip install opentelemetry-sdk==1.26.0
pip install --no-deps -e .

python examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
python examples/data_preprocess/gsm8k_custom.py