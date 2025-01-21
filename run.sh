curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --score-threshold 0.5 --train-batch-size 1 --val-batch-size 1 --epochs 5