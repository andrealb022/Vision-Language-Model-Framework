import json
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
import matplotlib.pyplot as plt
from .face_dataset import FaceDataset

AGE_CLASS_NAMES = ["0-2","3-9","10-19","20-29","30-39","40-49","50-59","60-69","70+"]

class Evaluator:
    """
    Classe statica per valutare le predizioni di modelli su diversi dataset.
    Supporta salvataggio dei risultati, calcolo di metriche e generazione di confusion matrix.
    """

    @staticmethod
    def evaluate(preds, gts, output_dir, dataset_name, age_mode: str = "auto"):
        """
        Metodo principale per valutare le predizioni in base al tipo di dataset.

        Args:
            preds (list): Lista di predizioni (dict per esempio).
            gts (list): Lista di ground truth corrispondenti (dict).
            output_dir (str or Path): Directory dove salvare i risultati (relativa al package evaluator).
            dataset_name (str): "MiviaPar" oppure uno tra quelli di FaceDataset.
            age_mode (str): "auto" | "classification" | "regression".
                            - auto: decide in base ai valori (0..8 interi -> classificazione, altrimenti regressione).
        """
        output_dir = Path(__file__).parent.resolve() / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        Evaluator._save_json(preds, output_dir / "preds.json")
        Evaluator._save_json(gts, output_dir / "gts.json")

        if dataset_name == "MiviaPar":
            Evaluator._evaluate_mivia_par(preds, gts, output_dir)
            print(f"[MIVIA PAR] Results saved in {output_dir}")
        elif dataset_name in FaceDataset.get_available_datasets():
            Evaluator._evaluate_face_dataset(preds, gts, output_dir, age_mode=age_mode)
            print(f"[FACE DATASET] Results saved in {output_dir}")
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    @staticmethod
    def _save_json(data, path):
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"[Errore] Salvataggio JSON fallito in {path}: {e}")

    @staticmethod
    def _plot_confusion_matrix(cm, labels, task, acc, output_path):
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar()
        plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45, ha="right")
        plt.yticks(ticks=range(len(labels)), labels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{task.upper()} - Acc: {acc:.4f}")
        thresh = cm.max() / 2.0 if cm.size else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def _evaluate_mivia_par(preds, gts, output_dir):
        metrics = {}
        accuracies = []
        tasks = preds[0].keys() if preds else []

        for task in tasks:
            y_true, y_pred = [], []
            for p, g in zip(preds, gts):
                if task in p and g.get(task, -1) != -1:
                    y_true.append(g[task])
                    y_pred.append(p[task])
            if not y_true:
                continue

            acc = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            labels = sorted(set(y_true + y_pred))
            accuracies.append(acc)

            metrics[task] = {
                "accuracy": acc,
                "labels": labels
            }

            Evaluator._plot_confusion_matrix(cm, labels, task, acc, output_dir / f"confusion_matrix_{task}.png")

        metrics["average_accuracy"] = sum(accuracies) / len(accuracies) if accuracies else None
        Evaluator._save_json(metrics, output_dir / "metrics.json")

    @staticmethod
    def _infer_age_mode_from_values(y_true_age, y_pred_age):
        """
        Se tutti i valori validi sono interi tra 0..8 → classification, altrimenti regression.
        """
        vals = [v for v in (y_true_age + y_pred_age) if v is not None]
        if not vals:
            return "regression"  # fallback, ma non cambierà nulla se non ci sono età
        try:
            as_int = [int(v) for v in vals]
        except (TypeError, ValueError):
            return "regression"
        if all((0 <= v <= 8) for v in as_int) and all(float(v).is_integer() for v in vals):
            return "classification"
        return "regression"

    @staticmethod
    def _evaluate_face_dataset(preds, gts, output_dir, age_mode: str = "auto"):
        """
        Valuta le predizioni per un dataset facciale.
        - Accuratezza + confusion matrix per gender, emotion, ethnicity.
        - AGE:
            * classification: accuratezza + confusion matrix sulle 9 classi
            * regression: MAE
        """
        metrics = {}
        accuracies = []
        tasks_accuracy = ["gender", "ethnicity", "emotion"]

        # --- Task di classificazione standard ---
        for task in tasks_accuracy:
            y_true, y_pred = [], []
            for p, g in zip(preds, gts):
                if task in p and g.get(task, -1) != -1:
                    y_true.append(g[task])
                    y_pred.append(p[task])
            if y_true:
                acc = accuracy_score(y_true, y_pred)
                cm = confusion_matrix(y_true, y_pred)
                labels = sorted(set(y_true + y_pred))
                metrics[task] = {
                    "accuracy": acc,
                    "labels": labels
                }
                accuracies.append(acc)
                Evaluator._plot_confusion_matrix(cm, labels, task, acc, output_dir / f"confusion_matrix_{task}.png")

        # --- AGE: classificazione o regressione ---
        y_true_age, y_pred_age = [], []
        for p, g in zip(preds, gts):
            if "age" in p and g.get("age", -1) != -1:
                y_true_age.append(g["age"])
                y_pred_age.append(p["age"])

        # Se non ci sono età, niente metrica età
        if y_true_age:
            if age_mode == "auto":
                decided = Evaluator._infer_age_mode_from_values(y_true_age, y_pred_age)
            else:
                decided = age_mode.lower()
                if decided not in {"classification", "regression"}:
                    decided = "regression"

            if decided == "classification":
                # Indici 0..8
                y_true_cls = [int(v) for v in y_true_age]
                y_pred_cls = [int(v) for v in y_pred_age]
                acc = accuracy_score(y_true_cls, y_pred_cls)
                cm = confusion_matrix(y_true_cls, y_pred_cls, labels=list(range(9)))
                metrics["age"] = {
                    "mode": "classification",
                    "accuracy": acc,
                    "labels": AGE_CLASS_NAMES
                }
                accuracies.append(acc)
                Evaluator._plot_confusion_matrix(
                    cm, AGE_CLASS_NAMES, "age", acc, output_dir / "confusion_matrix_age.png"
                )
            else:
                # Regressione: MAE
                y_true_reg = [float(v) for v in y_true_age]
                y_pred_reg = [float(v) for v in y_pred_age]
                mae = mean_absolute_error(y_true_reg, y_pred_reg)
                metrics["age"] = {
                    "mode": "regression",
                    "mae": mae
                }

        metrics["average_accuracy"] = sum(accuracies) / len(accuracies) if accuracies else None
        Evaluator._save_json(metrics, output_dir / "metrics.json")