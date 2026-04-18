import argparse
from core.config import Config
from core.utils import exit_with_exception, print_warning
from latency_estimation.dataset import prune_dataset, load_dataset
from latency_estimation.model_provider import ModelProvider
from latency_estimation.model import TsneItem
from ..providers.path_provider import PathProvider

SCALE = 1.0 # FIXME
DATASET_ID = 'FIXME' # FIXME

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Run t-SNE analysis on the dataset for a trained QPP-Net model.')
    add_args(parser)
    args = parser.parse_args(rawArgs)

    config = Config.load()

    try:
        run(config, args.checkpoint)
    except Exception as e:
        exit_with_exception(e)

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--checkpoint', '-c', type=str, help='Path to model checkpoint.')

def run(config: Config, checkpoint: str):
    pp = PathProvider(config)
    mp = ModelProvider(config)

    dataset = load_dataset(pp.dataset(DATASET_ID))

    model = mp.load_model(checkpoint)
    val_dataset = prune_dataset(dataset, model)

    tsne_items = list[TsneItem]()
    for item in val_dataset:
        tsne_items.extend(model.get_tsne_data(item.plan))

    tsne_by_operator = dict[str, list[TsneItem]]()
    for item in tsne_items:
        key = item.operator.key()
        if key not in tsne_by_operator:
            tsne_by_operator[key] = []
        tsne_by_operator[key].append(item)

    for items in tsne_by_operator.values():
        try:
            tsne_for_operator(items)
        except Exception as e:
            print_warning('Could not generate t-SNE plot for operator.', e)

def tsne_for_operator(items: list[TsneItem]):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    operator = items[0].operator

    print(f'\nOperator: {operator.key()}, Items: {len(items)}')
    features = np.array([item.features for item in items])
    estimated = np.array([item.estimated for item in items])
    extracted = np.array([item.extracted for item in items])
    difference = extracted - estimated

    perplexity = min(30, (len(features) - 1) // 3)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    x = StandardScaler().fit_transform(features)

    if np.var(x) < 1e-8:
        # There is no variance in the features, t-SNE will fail. Skip in this case.
        print_warning('No variance in features, skipping t-SNE for this operator.')
        return

    X_2d = tsne.fit_transform(x)

    plt.figure(figsize=(16, 5))
    plt.suptitle(f't-SNE of input features for operator: {operator.key()}')

    plt.subplot(1, 3, 1)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=estimated)
    plt.colorbar()
    plt.title("Estimated latency")

    plt.subplot(1, 3, 2)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=extracted)
    plt.colorbar()
    plt.title("Extracted latency")

    plt.subplot(1, 3, 3)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=difference)
    plt.colorbar()
    plt.title("Difference")

    plt.show()

if __name__ == '__main__':
    main()
