import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_tsne(
    embeddings,
    labels,
    image_paths=None,
    show=True,
    title="t-SNE plot of embeddings",
    teams=None,
    n_images=10,
    **kwargs,
):
    if teams is None:
        teams = [0, 1]
    if image_paths is not None:
        assert len(image_paths) == len(
            embeddings
        ), f"image_paths length {len(image_paths)} does not match embeddings length {len(embeddings)}"

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # generate t-SNE embeddings
    tsne = TSNE(n_components=2, **kwargs)
    tsne_results = tsne.fit_transform(embeddings)

    # plot t-SNE results with ground truth labels and color by team
    fig, ax = plt.subplots(figsize=(10, 10))

    for team in teams:
        ax.scatter(
            tsne_results[labels == team, 0],
            tsne_results[labels == team, 1],
            c=f"C{team}",
            label=f"Team {team}",
        )

    # set plot x and y limits
    ax.set_xlim(tsne_results[:, 0].min(), tsne_results[:, 0].max() + 1)
    ax.set_ylim(tsne_results[:, 1].min(), tsne_results[:, 1].max() + 1)
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    # plot some images on top of the t-SNE plot
    scale_factor = tsne_results[:, 0].max() - tsne_results[:, 0].min()
    scale_factor /= 30

    if image_paths is not None:
        for i in np.random.choice(range(len(image_paths)), size=n_images):
            image = cv2.imread(image_paths[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # plot image at the embedding location
            bbox = np.array(
                (
                    tsne_results[i, 0] - scale_factor / 2,
                    tsne_results[i, 0] + scale_factor / 2,
                    tsne_results[i, 1] - scale_factor / 2,
                    tsne_results[i, 1] + scale_factor / 2,
                )
            )
            ax.imshow(
                image,
                extent=bbox,
                zorder=1,
            )
    plt.legend()

    if show:
        plt.show()
    else:
        return fig, ax
