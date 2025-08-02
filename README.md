# Vision-Language-Models Framework
Questo repository contiene un **framework modulare e estensibile** per l'inferenza, la valutazione e la comparazione di modelli Vision-Language (VLM), come BLIP-2, LLaVA e PaLI-Gemma.

Supporta:
- Caricamento automatico di dataset standardizzati
- Parsing testuale delle predizioni in etichette strutturate
- Calcolo di metriche (accuracy, MAE, confusion matrix)
- Integrazione plug-and-play di nuovi modelli o dataset
---

## ⚙️ Installazione

### 1. Usare Conda (consigliato)

```bash
conda env create -f environment.yml
conda activate framework
```

### 2. Alternativa: pip (manuale)

```bash
pip install -r requirements.txt
```
---

## 🚀 Esecuzione

### Inferenzare un dataset con un VLM:

```bash
python prompt_inference.py \
    --model_name llava \
    --model_id llava-hf/llava-1.5-7b-hf \
    --dataset_name MiviaPar \
    --output_dir output/llava \
    --quantization fp32 \
    --train False
```
---

## 🧠 Modelli supportati

| Nome modello | ID HuggingFace                        | File              |
|--------------|----------------------------------------|-------------------|
| `blip2`      | `Salesforce/blip2-opt-6.7b`            | `blip2.py`        |
| `llava`      | `llava-hf/llava-1.5-7b-hf`             | `llava.py`        |
| `paligemma`  | `google/paligemma-3b-mix-224`          | `paligemma.py`    |

Puoi registrarne altri via `model_factory.py`.

---

## 📚 Dataset supportati

- **MiviaPar** (gender, color, bag, hat)
- **CelebA_HQ**, **FairFace**, **RAF-DB**, **UTKFace**, ecc. via `FaceDataset`

I dataset devono essere organizzati come segue:

```
~/datasets_with_standard_labels/
└── MiviaPar/
    └── test/
        ├── images/
        └── labels.csv
```

---

## 📈 Valutazione

Il framework calcola automaticamente:
- Accuracy per attributi categoriali (es. gender, emotion, ethnicity)
- MAE per età
- Confusion Matrix (salvate in PNG)
---

## 🧩 Estensibilità

Per aggiungere un nuovo modello:
1. Crea una sottoclasse in `models/` che estende `VLMModel`
2. Implementa `_load()` e `generate_text(...)`
3. Registra il modello in `model_factory.py`

Per aggiungere un nuovo dataset:
1. Estendi `BaseDataset`
2. Implementa `_load_labels()` e `get_labels_from_text_output()`
3. Registra il nome nel metodo in `dataset_factory.py`

---

## 🛠 Requisiti
- Python 3.10+
- `transformers`, `torch`, `PIL`, `matplotlib`, `scikit-learn`, `pandas`
---

## 📌 Note
- Per usare `PaLI-Gemma`, serve autenticazione Hugging Face.
- I modelli sono valutati in modalità zero-shot, senza fine-tuning.
- L'output testuale viene convertito in etichette tramite parser specifici per dataset.
---