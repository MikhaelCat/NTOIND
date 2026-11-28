### 1. Соответствие требованиям регламента

* **Полная воспроизводимость:** пайплайн подготовки, обучения и предсказания полностью автоматизирован и детерминирован.
* **Запуск одной командой:** `python main.py` — сам обучает все модели (если нет предсказаний) и генерирует `submission.csv`.
* **Время работы:**
  - Полный цикл переобучения ансамбля на GPU: **< 4 часов** на 1x NVIDIA T4 (16 GB VRAM) и 12 GB RAM.
* **Работа без сети (офлайн):** решение работает автономно при наличии локальных весов в `models/`. Если папки `models/` нет — базовые модели скачиваются с Hugging Face автоматически.
* **Фиксация сидов:** все генераторы псевдослучайных чисел зафиксированы (`seed=42`, `torch.manual_seed`, `np.random.seed`, детерминированный `GroupKFold`).

---

### 2. Описание архитектуры решения

1. **Multi-Peer 5-Way Transformer Ensemble:**
   * `mdeberta-v3-base-xnli` (512 токенов, peer-context view, вес 0.35)
   * `xlm-roberta-large-xnli` (512 токенов, peer-context view, вес 0.20)
   * `cointegrated/rubert-base-cased-nli-threeway` (512 токенов, peer-context view, вес 0.10)
   * `mdeberta-v3-base-xnli` (512 токенов, full-sequence baseline, вес 0.20)
   * `xlm-roberta-large-xnli` (512 токенов, full-sequence baseline, вес 0.15)
   * Обучение на 5 фолдах `GroupKFold` (группировка по тексту поста `text`), полностью исключающая утечки данных между фолдами.
2. **Контекст соседнего утверждения (Peer-Context):**
   * Для каждого утверждения в рамках одного поста подбирается наиболее похожее соседнее утверждение без использования его метки. Это позволяет модели напрямую сравнивать тонкие мутации формулировок (изменение чисел, дат, отрицаний).
3. **Совместное структурированное декодирование (Joint Structured Decode):**
   * Все утверждения одного поста декодируются совместно методом полного перебора допустимых конфигураций $3^N$ ($N \le 6$) с байесовским композиционным априором и парной моделью `HistGradientBoostingClassifier`, обученной строго на обучающей выборке.

---

### 3. Структура директории `solution/`

```
solution/
├── main.py                          # Главная точка входа: авто-обучение + сборка сабмита
├── train.py                         # Обучение одной модели на GPU (вызывается из main.py)
├── download_models.py               # Опциональный загрузчик базовых моделей с HuggingFace
├── strict_structured_decode.py      # Модуль структурированного декодирования
├── requirements.txt                 # Зависимости Python
└── README.md                        # Документация
```

> Папки `gpu_full_*/` и файлы `*.npy` / `*.npz` **не прикладываются** — они генерируются автоматически при первом запуске `python main.py`.

---

### 4. Запуск одной командой

Убедитесь, что в папке рядом с `solution/` (или в самой `solution/`) есть `train.csv`, `test.csv`, `sample_submission.csv`.

```bash
pip install -r requirements.txt
python main.py
```

При первом запуске:
1. Скачиваются базовые веса моделей с Hugging Face (если нет локальной папки `models/`).
2. Обучаются компоненты ансамбля на GPU в пределах регламентного лимита по времени.
3. Сохраняются сгенерированные предсказания в `gpu_full_*/probabilities.npz`.
4. Генерируется валидный `submission.csv`.

При повторном запуске (когда файлы `.npz` уже сгенерированы в ходе пайплайна) — сразу выполняется инференс и сборка `submission.csv` (< 30 секунд).

```bash
python main.py --train    # Принудительно переобучить все модели заново
python main.py --no-train # Только инференс; упадёт с ошибкой, если .npz нет
```

---

### 5. Полное переобучение моделей с нуля (GPU, < 4 часов на NVIDIA T4)

```bash
python main.py --train
```

Или каждую конфигурацию вручную:
```bash
python train.py --model mdeberta --view peer_strong --batch-size 16 --epochs 3 --max-length 512 --lr 1.5e-5 --out gpu_full_mdeberta_peerstrong512
python train.py --model xlm     --view peer_strong --batch-size  8 --epochs 3 --max-length 512 --lr 1.5e-5 --out gpu_full_xlm_peerstrong512
python train.py --model rubert  --view peer_strong --batch-size 16 --epochs 3 --max-length 512 --lr 2.0e-5 --out gpu_full_rubert_peerstrong512
python train.py --model mdeberta --view full        --batch-size 16 --epochs 3 --max-length 512 --lr 1.5e-5 --out gpu_full_mdeberta_full512
python train.py --model xlm     --view full        --batch-size  8 --epochs 3 --max-length 512 --lr 1.5e-5 --out gpu_full_xlm_full512
```
