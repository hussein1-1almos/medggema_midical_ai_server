# Run from the project root in an activated virtual environment.
# You must first accept the Health AI Developer Foundations terms and request
# access to google/medgemma-1.5-4b-it on Hugging Face.
$ErrorActionPreference = "Stop"
python -m pip install -U "huggingface_hub[cli]"
hf auth login
hf download google/medgemma-1.5-4b-it --local-dir models/medgemma-1.5-4b-it
Write-Host "Model download complete. Set MEDGEMMA_MODEL_DIR=models/medgemma-1.5-4b-it in .env"
