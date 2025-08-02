import json
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
import matplotlib.pyplot as plt
from datasets.face_dataset import FaceDataset

class Evaluator:
    """
    Classe statica per valutare le predizioni di modelli su diversi dataset.
    Supporta salvataggio dei risultati, calcolo di metriche e generazione di confusion matrix.
    """

    @staticmethod
    def evaluate(preds, gts, output_dir, dataset_name):
        """
        Metodo principale per valutare le predizioni in base al tipo di dataset.

        Args:
            preds (list): Lista di predizioni (dict per esempio).
            gts (list): Lista di ground truth corrispondenti (dict).
            output_dir (str or Path): Directory dove salvare i risultati.
            dataset_name (str): Nome del dataset ("MiviaPar" o uno tra quelli di FaceDataset).
        """
        output_dir = Path(__file__).parent.resolve() / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        Evaluator._save_json(preds, output_dir / "preds.json")
        Evaluator._save_json(gts, output_dir / "gts.json")

        if dataset_name == "MiviaPar":
            Evaluator._evaluate_mivia_par(preds, gts, output_dir)
            print(f"[MIVIA PAR] Results saved in {output_dir}")
        elif dataset_name in FaceDataset.get_available_datasets():
            Evaluator._evaluate_face_dataset(preds, gts, output_dir)
            print(f"[FACE DATASET] Results saved in {output_dir}")
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    @staticmethod
    def _save_json(data, path):
        """
        Salva dati in formato JSON in un file.

        Args:
            data (Any): Dati serializzabili in JSON.
            path (Path): Percorso del file di destinazione.
        """
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"[Errore] Salvataggio JSON fallito in {path}: {e}")

    @staticmethod
    def _plot_confusion_matrix(cm, labels, task, acc, output_path):
        """
        Genera e salva una confusion matrix.

        Args:
            cm (np.ndarray): Matrice di confusione.
            labels (list): Etichette delle classi.
            task (str): Nome del task (per titolo del grafico).
            acc (float): Accuratezza del task.
            output_path (Path): Percorso del file immagine.
        """
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar()
        plt.xticks(ticks=range(len(labels)), labels=labels)
        plt.yticks(ticks=range(len(labels)), labels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{task.upper()} - Acc: {acc:.4f}")
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def _evaluate_mivia_par(preds, gts, output_dir):
        """
        Valuta le predizioni per il dataset MIVIA PAR.
        Calcola accuratezza per ogni task e media, salva confusion matrix e metriche.

        Args:
            preds (list): Lista di predizioni (dict).
            gts (list): Lista di ground truth (dict).
            output_dir (Path): Directory di output.
        """
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
    def _evaluate_face_dataset(preds, gts, output_dir):
        """
        Valuta le predizioni per un dataset facciale.
        Calcola accuratezza per gender, emotion, ethnicity e MAE per age.

        Args:
            preds (list): Lista di predizioni (dict).
            gts (list): Lista di ground truth (dict).
            output_dir (Path): Directory di output.
        """
        metrics = {}
        accuracies = []
        tasks_accuracy = ["gender", "ethnicity", "emotion"]
        task_mae = "age"

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

        # MAE per l'età
        y_true_age, y_pred_age = [], []
        for p, g in zip(preds, gts):
            if task_mae in p and g.get(task_mae, -1) != -1:
                y_true_age.append(g[task_mae])
                y_pred_age.append(p[task_mae])
        if y_true_age:
            mae = mean_absolute_error(y_true_age, y_pred_age)
            metrics[task_mae] = {"mae": mae}

        metrics["average_accuracy"] = sum(accuracies) / len(accuracies) if accuracies else None
        Evaluator._save_json(metrics, output_dir / "metrics.json")
