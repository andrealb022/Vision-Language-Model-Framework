"""
Preprocess dei dataset con due funzionalità principali:

1) Creazione dello split di validazione (val/) se mancante — move-only
   - Per ogni dataset sotto ~/datasets_with_standard_labels/ che contiene train/:
     * se val/ NON esiste → crea uno split 80/20 e SPOSTA fisicamente i file da train/ a val/
       (mai copia).
     * se val/ esiste → non fa nulla per quello split.
     * Per il dataset 'VggFace2-Train', se presente la colonna 'Identity', lo split è per identità;
       altrimenti è per riga (immagine).
     * I percorsi scritti nei CSV (train/labels.csv e val/labels.csv) NON includono l’estensione.

2) Calcolo dei conteggi per classe sullo split train/
   - Per ogni dataset (che abbia train/labels.csv), calcola i conteggi per:
       Gender (0/1), Ethnicity (0..3), Facial Emotion (0..6), Age (bin 0..8).
     I risultati sono salvati in train/class_counts.json. I valori '-1' sono esclusi dai conteggi.
"""
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================================================================
# Configurazione
# =============================================================================

BASE_DIR = Path("~/datasets_with_standard_labels/").expanduser()
IMAGES_DIR = "images"
LABELS_FILE = "labels.csv"
VAL_RATIO = 0.2  # 20% validation

AGE_LABELS = {
    "0-2": 0, "3-9": 1, "10-19": 2, "20-29": 3, "30-39": 4,
    "40-49": 5, "50-59": 6, "60-69": 7, "70+": 8,
}
# Soglie superiori (incluse) per mappare un'età float alla classe 0..8
AGE_BOUNDS = [2, 9, 19, 29, 39, 49, 59, 69, float("inf")]

# =============================================================================
# Utility filesystem / path
# =============================================================================

def ensure_dir(path: Path) -> None:
    """Crea la directory (e genitori) se non esiste."""
    path.mkdir(parents=True, exist_ok=True)


def extract_rel_inside_images(raw_path: str, dataset_name: str, split: str) -> Path:
    """
    Dato un valore (libero) della colonna 'Path' del CSV, restituisce SEMPRE la
    parte relativa interna a 'images/' (ad es. 'id123/0001').

    Gestisce:
    - path con prefisso logico 'datasets_with_standard_labels/<dataset>/<split>/images/...'
    - path assoluti che contengono '/images/'
    - path già relativi
    - slash/backslash
    """
    s = str(raw_path).strip().replace("\\", "/")
    key = "datasets_with_standard_labels/"

    # Caso: prefisso logico presente
    if key in s:
        parts = s.split("/")
        if "images" in parts:
            i_img = parts.index("images")
            return Path(*parts[i_img + 1:])
        return Path(parts[-1])

    # Caso: contiene già '/images/'
    if "/images/" in s:
        return Path(s.split("/images/", 1)[1])

    # Caso: path assoluto → cerca 'images'
    p = Path(s)
    if p.is_absolute():
        parts = [*p.parts]
        parts_lower = [pp.lower() for pp in parts]
        if "images" in parts_lower:
            i_img = parts_lower.index("images")
            return Path(*parts[i_img + 1:])
        return Path(p.name)

    # Caso: già relativo (probabilmente sotto images/)
    return Path(s)

def resolve_src_from_train_images(train_images_dir: Path, rel_inside_images: Path) -> Optional[Path]:
    """
    Trova il file in train/images dato un path relativo interno (senza o con estensione).
    Se manca il suffisso, prova .jpg, .jpeg, .png in quest’ordine.
    Ritorna il path esistente oppure None.
    """
    candidate = train_images_dir / rel_inside_images
    if candidate.exists():
        return candidate

    if candidate.suffix == "":
        for ext in (".jpg", ".jpeg", ".png"):
            c = candidate.with_suffix(ext)
            if c.exists():
                return c

    return None

def build_csv_path_for_split(dataset_name: str, split: str, rel_noext_inside_images: Path) -> str:
    """
    Costruisce la stringa da scrivere in CSV (colonna 'Path'), sempre SENZA estensione, in formato:
        datasets_with_standard_labels\<Dataset>\<split>\images\<rel_noext>
    (usa backslash).
    """
    rel_norm = str(rel_noext_inside_images).replace("/", "\\")
    return f"datasets_with_standard_labels\\{dataset_name}\\{split}\\images\\{rel_norm}"


def path_without_suffix(p: Path) -> Path:
    """Rimuove l'ultima estensione dal path mantenendo le sottocartelle."""
    return p.with_suffix("")

# =============================================================================
# Split helpers
# =============================================================================

def random_row_split(n_rows: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split per riga (immagine): restituisce maschere booleane (train_mask, val_mask).
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows)
    rng.shuffle(idx)
    k = max(1, int(round(n_rows * val_ratio)))
    val_idx = set(idx[:k])
    val_mask = np.array([i in val_idx for i in range(n_rows)], dtype=bool)
    train_mask = ~val_mask
    return train_mask, val_mask


def groupwise_split(groups: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split per gruppi (identità): seleziona ~val_ratio dei gruppi e assegna TUTTE le loro righe a val.
    Restituisce (train_mask, val_mask).
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    k = max(1, int(round(len(uniq) * val_ratio)))
    val_groups = set(uniq[:k])
    val_mask = np.isin(groups, list(val_groups))
    train_mask = ~val_mask
    return train_mask, val_mask

# =============================================================================
# CSV helpers
# =============================================================================

def load_csv_with_header(csv_path: Path) -> Tuple[pd.DataFrame, str, Optional[str]]:
    """
    Legge labels.csv assumendo un HEADER presente.
    Ritorna: (DataFrame normalizzato, nome_colonna_Path, nome_colonna_Identity_o_None).
    """
    df = pd.read_csv(csv_path, header=0)
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    lower = [c.lower() for c in cols]
    if "path" not in lower:
        raise ValueError(f"Nel CSV '{csv_path}' manca la colonna 'Path' (con header).")

    path_col = cols[lower.index("path")]
    ident_col = cols[lower.index("identity")] if "identity" in lower else None
    return df, path_col, ident_col


def write_csv(df: pd.DataFrame, out_csv: Path) -> None:
    """Scrive un DataFrame su CSV mantenendo l’header e senza indice."""
    df.to_csv(out_csv, index=False)

# =============================================================================
# Feature 1: crea val/ se mancante (move-only)
# =============================================================================

def create_val_split_if_missing(dataset_dir: Path, seed: int, verbose: bool) -> bool:
    """
    Se 'val/' NON esiste:
      - crea split 80/20 (per riga; per identità SOLO se dataset = 'VggFace2-Train' e colonna 'Identity' presente);
      - SPOSTA i file in val/images;
      - riscrive train/labels.csv e crea val/labels.csv (colonna Path SENZA estensione).
    Se 'val/' esiste, non fa nulla.

    Ritorna True se il dataset è stato considerato (split creato o già presente), False se non standard/mancante.
    """
    dataset_name = dataset_dir.name
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"

    train_images = train_dir / IMAGES_DIR
    train_labels = train_dir / LABELS_FILE
    if not train_images.exists() or not train_labels.exists():
        return False  # dataset non standard / train incompleto

    if val_dir.exists():
        if verbose:
            print(f"[SKIP] {dataset_name}: 'val/' esiste già → nessuna modifica")
        return True

    # Carica CSV train
    df, path_col, ident_col = load_csv_with_header(train_labels)

    # Modalità di split
    split_mode = "row"
    if dataset_name == "VggFace2-Train" and ident_col is not None:
        split_mode = "identity"

    # Maschere train/val
    if split_mode == "identity":
        groups = df[ident_col].astype(str).str.strip().values
        tr_mask, va_mask = groupwise_split(groups, VAL_RATIO, seed)
    else:
        tr_mask, va_mask = random_row_split(len(df), VAL_RATIO, seed)

    df_train_new = df.loc[tr_mask].copy()
    df_val = df.loc[va_mask].copy()

    # Crea cartelle val/images
    val_images = val_dir / IMAGES_DIR
    ensure_dir(val_images)

    # Sposta i file da train → val
    moved = 0
    for _, row in tqdm(df_val.iterrows(), total=len(df_val), desc=f"[{dataset_name}] moving to val"):
        rel_inside = extract_rel_inside_images(row[path_col], dataset_name, "train")
        src = resolve_src_from_train_images(train_images, rel_inside)
        if src is None:
            raise FileNotFoundError(
                f"File non trovato in train/images: {train_images / rel_inside} "
                f"(provate estensioni .jpg/.jpeg/.png)"
            )

        # Per il filesystem: se il nome relativo non ha suffisso, usa quello del file sorgente.
        rel_fs = rel_inside if rel_inside.suffix != "" else rel_inside.with_suffix(src.suffix)

        dst = val_images / rel_fs
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved += 1

    # Aggiorna la colonna 'Path' per CSV (scrittura SENZA estensione)
    def remap_df_paths(df_split: pd.DataFrame, split_name: str) -> pd.DataFrame:
        out = df_split.copy()
        new_paths = []
        for raw in out[path_col].astype(str):
            rel_inside = extract_rel_inside_images(raw, dataset_name, "train")
            rel_noext = path_without_suffix(rel_inside)  # CSV senza suffisso
            new_paths.append(build_csv_path_for_split(dataset_name, split_name, rel_noext))
        out[path_col] = new_paths
        return out

    df_train_new = remap_df_paths(df_train_new, "train")
    df_val = remap_df_paths(df_val, "val")

    # Scrivi i CSV aggiornati
    write_csv(df_train_new, train_dir / LABELS_FILE)
    write_csv(df_val, val_dir / LABELS_FILE)

    if verbose:
        print(
            f"[OK] {dataset_name}: split={'identity' if split_mode=='identity' else 'row'}, "
            f"train->{len(df_train_new)}, val->{len(df_val)} (spostati: {moved})"
        )
    return True

# =============================================================================
# Feature 2: conteggi per classe (train/)
# =============================================================================

def age_float_to_bin(age_val: float) -> int:
    """Mappa un valore float d'età nella classe 0..8 usando AGE_BOUNDS; -1 se negativo."""
    if age_val < 0:
        return -1
    for idx, upper in enumerate(AGE_BOUNDS):
        if age_val <= upper:
            return idx
    return -1


def age_to_class(v) -> int:
    """
    Converte un valore Age in classe 0..8:
    - se stringa 'A-B' usa la tabella AGE_LABELS;
    - altrimenti prova a interpretarlo come numero (float → bin).
    Ritorna -1 se non interpretabile.
    """
    if isinstance(v, str):
        s = v.strip()
        if s in AGE_LABELS:
            return AGE_LABELS[s]
        try:
            f = float(s)
        except Exception:
            return -1
        return age_float_to_bin(f)

    try:
        if isinstance(v, (int, np.integer)) and 0 <= int(v) <= 8:
            return int(v)
        f = float(v)
        return age_float_to_bin(f)
    except Exception:
        return -1


def count_classes_for_train(dataset_dir: Path, verbose: bool) -> Optional[Dict[str, Dict[str, int]]]:
    """
    Legge train/labels.csv e produce i conteggi per classe delle colonne standard:
      - Gender (0/1)
      - Ethnicity (0..3)
      - Facial Emotion (0..6)
      - Age (mappata sui bin 0..8)
    Esclude i valori '-1'. Salva il risultato in train/class_counts.json.
    """
    train_dir = dataset_dir / "train"
    labels_csv = train_dir / LABELS_FILE
    if not train_dir.exists() or not labels_csv.exists():
        return None

    df, _, _ = load_csv_with_header(labels_csv)
    cols_lower = {c.lower(): c for c in df.columns}
    counts: Dict[str, Dict[str, int]] = {}

    # Gender
    if "gender" in cols_lower:
        col = cols_lower["gender"]
        s = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        s = s[s >= 0]
        vc = s.value_counts().sort_index()
        counts["gender"] = {str(int(k)): int(v) for k, v in vc.items()}

    # Ethnicity
    if "ethnicity" in cols_lower:
        col = cols_lower["ethnicity"]
        s = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        s = s[s >= 0]
        vc = s.value_counts().sort_index()
        counts["ethnicity"] = {str(int(k)): int(v) for k, v in vc.items()}

    # Facial Emotion
    if "facial emotion" in cols_lower:
        col = cols_lower["facial emotion"]
        s = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        s = s[s >= 0]
        vc = s.value_counts().sort_index()
        counts["emotion"] = {str(int(k)): int(v) for k, v in vc.items()}

    # Age (binning)
    if "age" in cols_lower:
        col = cols_lower["age"]
        classes = df[col].apply(age_to_class)
        classes = classes[classes >= 0]
        vc = classes.value_counts().sort_index()
        counts["age"] = {str(int(k)): int(v) for k, v in vc.items()}

    # Salvataggio
    out_path = train_dir / "class_counts.json"
    out_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")

    if verbose:
        print(f"[OK] {dataset_dir.name}: salvato {out_path}")

    return counts

# =============================================================================
# Main
# =============================================================================

def main():
    """
    Esegue le due funzionalità principali del preprocess:
      1) crea lo split di validazione 80/20 (val/) se mancante — move-only;
      2) calcola e salva i conteggi per classe su train/.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess dei dataset: crea val 80/20 (se mancante, move-only) e calcola class counts su train. "
            "I CSV di output NON includono l'estensione nel campo Path."
        )
    )
    parser.add_argument(
        "--base",
        type=str,
        default=str(BASE_DIR),
        help="Cartella base (default: ~/datasets_with_standard_labels/)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed RNG per lo split")
    parser.add_argument("--verbose", action="store_true", help="Log dettagliato")
    args = parser.parse_args()

    base = Path(args.base).expanduser()
    if not base.exists():
        raise FileNotFoundError(f"Base non trovata: {base}")

    datasets = [d for d in base.iterdir() if d.is_dir()]
    processed_split = 0
    processed_counts = 0

    for ds_dir in sorted(datasets):
        train_dir = ds_dir / "train"
        if not train_dir.exists():
            continue

        # (1) Crea val/ se mancante (move-only)
        try:
            ok_split = create_val_split_if_missing(ds_dir, seed=args.seed, verbose=args.verbose)
            if ok_split:
                processed_split += 1
        except Exception as e:
            print(f"[ERR] split {ds_dir.name}: {e}")

        # (2) Calcola conteggi per classe su train/
        try:
            res = count_classes_for_train(ds_dir, verbose=args.verbose)
            if res is not None:
                processed_counts += 1
        except Exception as e:
            print(f"[ERR] counts {ds_dir.name}: {e}")

    print(f"[DONE] Split creati/verificati: {processed_split} | Counts calcolati: {processed_counts}")

if __name__ == "__main__":
    main()