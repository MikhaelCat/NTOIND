"""Download required Hugging Face base models for training."""
import argparse
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELS = {
    'mdeberta': ('mdeberta-v3-base-xnli', 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'),
    'xlm': ('xlm-roberta-large-xnli', 'joeddav/xlm-roberta-large-xnli'),
    'rubert': ('rubert-base-cased-nli-threeway', 'cointegrated/rubert-base-cased-nli-threeway'),
}


def download_model(target_dir: Path, hf_name: str):
    has_weights = (
        (target_dir / 'config.json').exists()
        and (any(target_dir.glob('*.bin')) or any(target_dir.glob('*.safetensors')) or any(target_dir.glob('*.pt')))
    )
    if has_weights:
        print(f'[OK] Модель уже найдена: {target_dir}')
        return
    print(f'Скачивание {hf_name} в {target_dir}...')
    target_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    tokenizer.save_pretrained(target_dir)
    model = AutoModelForSequenceClassification.from_pretrained(hf_name)
    model.save_pretrained(target_dir)
    print(f'[Успех] Модель сохранена в {target_dir}')


def main():
    parser = argparse.ArgumentParser(description='Download base models from HuggingFace into models/ folder.')
    parser.add_argument('--models_dir', type=Path, default=None, help='Target directory for models (default: ../models or ./models)')
    args = parser.parse_args()

    pkg_dir = Path(__file__).resolve().parent
    models_dir = args.models_dir
    if models_dir is None:
        models_dir = pkg_dir.parent / 'models' if (pkg_dir.parent / 'train.csv').exists() else pkg_dir / 'models'

    models_dir.mkdir(parents=True, exist_ok=True)
    print(f'Директория для моделей: {models_dir}')

    for key, (folder_name, hf_id) in MODELS.items():
        download_model(models_dir / folder_name, hf_id)

    print('Все базовые модели проверены и готовы к обучению!')


if __name__ == '__main__':
    main()
