import pickle
import numpy as np
from matplotlib import pyplot
from training import Params, load_mnist, classify, NUMBER_OF_CLASSES
import logging as log

PLOTS_FOLDER = 'plots'
log.basicConfig(filename=PLOTS_FOLDER + '/out.md', filemode='w', format='%(message)s', level=log.INFO)

Metrics = dict[str, float]


def plot_cost(cost: list[float]) -> None:
    pyplot.clf()
    pyplot.plot(cost)
    pyplot.title(f'Train cost accuracy')
    pyplot.ylabel('Cost')
    pyplot.savefig(PLOTS_FOLDER + '/cost.png')


def compute_confusion_matrix(
        test_data: np.ndarray,
        test_target: np.ndarray,
        params: Params) -> tuple[np.ndarray, list[tuple[int, np.ndarray]], float]:
    matrix: np.ndarray = np.zeros(shape=(NUMBER_OF_CLASSES, NUMBER_OF_CLASSES))
    sample: list[tuple[int, np.ndarray]] = []

    correct: float = 0.0
    for iteration in range(len(test_data)):
        data: np.ndarray = test_data[iteration]

        output, confidence = classify(data, params)

        target: int = test_target[iteration]
        matrix[output][target] += 1
        sample.append((target, confidence))

        if output == target:
            correct += 1

    return matrix.astype(np.int32), sample, correct / len(test_data)


def compute_metrics(confusion_matrix: np.ndarray) -> list[Metrics]:
    model_metrics: list[Metrics] = []

    for cls in range(NUMBER_OF_CLASSES):
        tp: float = confusion_matrix[cls, cls]
        fp: float = confusion_matrix[cls, :].sum() - tp
        fn: float = confusion_matrix[:, cls].sum() - tp

        precision: float = tp / (tp + fp)
        recall: float = tp / (tp + fn)
        f1_score: float = 2 * precision * recall / (precision + recall)

        model_metrics.append(dict(
            precision=precision,
            recall=recall,
            f1_score=f1_score
        ))

    return model_metrics


def plot_metrics(model_metrics: list[Metrics]) -> None:
    precision: list[float] = [model_metrics[cls]['precision'] * 100 for cls in range(NUMBER_OF_CLASSES)]
    recall: list[float] = [model_metrics[cls]['recall'] * 100 for cls in range(NUMBER_OF_CLASSES)]
    f1_score: list[float] = [model_metrics[cls]['f1_score'] for cls in range(NUMBER_OF_CLASSES)]

    classes: list[int] = list(range(NUMBER_OF_CLASSES))

    pyplot.clf()
    pyplot.bar(classes, precision)
    pyplot.title('Precision')
    pyplot.xticks(classes)
    pyplot.yticks(ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                  labels=[f'{it}%' for it in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]])
    pyplot.savefig(PLOTS_FOLDER + '/precision.png')

    pyplot.clf()
    pyplot.bar(classes, recall)
    pyplot.title('Recall')
    pyplot.xticks(classes)
    pyplot.yticks(ticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                  labels=[f'{it}%' for it in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]])
    pyplot.savefig(PLOTS_FOLDER + '/recall.png')

    pyplot.clf()
    pyplot.bar(classes, f1_score)
    pyplot.title('F Score')
    pyplot.xticks(classes)
    pyplot.yticks(ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                  labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    pyplot.savefig(PLOTS_FOLDER + '/score.png')


def compute_class_examples_count(test_target: np.ndarray) -> list[int]:
    counts: list[int] = [0 for _ in range(NUMBER_OF_CLASSES)]
    for i in range(len(test_target)):
        counts[test_target[i]] += 1

    return counts


def roc_curve(
        cls: int,
        positive_examples_count: int,
        negative_examples_count: int,
        sample: list[tuple[int, np.ndarray]]) -> tuple[list[float], list[float], float]:
    fpr: list[float] = [0]
    tpr: list[float] = [0]
    auc: float = 0

    sample: list[tuple[int, np.ndarray]] = sorted(sample, key=lambda it: -it[1][cls])

    for i in range(len(sample)):
        if sample[i][0] == cls:
            fpr.append(fpr[-1])
            tpr.append(tpr[-1] + 1 / positive_examples_count)
        else:
            fpr.append(fpr[-1] + 1 / negative_examples_count)
            tpr.append(tpr[-1])
            auc += tpr[-1] / negative_examples_count

    log.info(f"- AUC for {cls}: {auc:.5f}")

    return fpr, tpr, auc


def plot_roc_curves(
        examples_count: list[int],
        sample: list[tuple[int, np.ndarray]],
        show: bool = False):
    pyplot.clf()
    aucs: list[float] = []
    for cls in range(NUMBER_OF_CLASSES):
        fpr, tpr, auc = roc_curve(cls, examples_count[cls], sum(examples_count) - examples_count[cls], sample)
        pyplot.plot(fpr, tpr)
        aucs.append(auc)

    pyplot.title('ROC Curve')
    pyplot.savefig(PLOTS_FOLDER + '/roc.png')
    if show:
        pyplot.show()


def as_binary(
        cls: int,
        sample: list[tuple[int, np.ndarray]]) -> tuple[list[int], list[float]]:
    truth_labels: list[int] = [1 if cls == it[0] else 0 for it in sample]
    scores: list[float] = [it[1][cls] for it in sample]

    return truth_labels, scores


def show_plots(file_name):
    with open(file_name + ".pkl", 'rb') as parameters_file:
        params, cost = pickle.load(parameters_file)

    cache_file_name: str = f'{file_name}-metrics.pkl'
    try:
        with open(cache_file_name, 'rb') as cache_file:
            confusion_matrix, sample, accuracy, examples_count = pickle.load(cache_file)
    except FileNotFoundError:
        _, (test_data, test_target) = load_mnist()
        confusion_matrix, sample, accuracy = compute_confusion_matrix(test_data, test_target, params)
        examples_count: list[int] = compute_class_examples_count(test_target)
        with open(cache_file_name, 'wb') as cache_file:
            pickle.dump([confusion_matrix, sample, accuracy, examples_count], cache_file)

    log.info("Accuracy: " + str(accuracy * 100) + " %")

    plot_cost(cost)
    plot_metrics(compute_metrics(confusion_matrix))
    plot_roc_curves(examples_count, sample)
