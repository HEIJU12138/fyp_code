from collections import Counter
import matplotlib.pyplot as plt
from imblearn.datasets import make_imbalance
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs

def ratio_func(y, multiplier, minority_class):
    target_stats = Counter(y)
    return {minority_class: int(multiplier * target_stats[minority_class])}

X, y = make_moons(n_samples=1000, shuffle=True, random_state=10,noise=0.1)
X = pd.DataFrame(X, columns=["feature 1", "feature 2"])
# ax = X.plot.scatter(
#     x="feature 1",
#     y="feature 2",
#     c=y,
#     colormap="viridis",
#     colorbar=False,
# )
# sns.despine(ax=ax, offset=10)



fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

X.plot.scatter(
    x="feature 1",
    y="feature 2",
    c=y,
    ax=axs[0, 0],
    colormap="viridis",
    colorbar=False,
)
axs[0, 0].set_title("Original set")
sns.despine(ax=axs[0, 0], offset=10)

multipliers = [0.9, 0.75, 0.5, 0.25, 0.1]
for ax, multiplier in zip(axs.ravel()[1:], multipliers):
    X_resampled, y_resampled = make_imbalance(
        X,
        y,
        sampling_strategy=ratio_func,
        **{"multiplier": multiplier, "minority_class": 1},
    )
    X_resampled.plot.scatter(
        x="feature 1",
        y="feature 2",
        c=y_resampled,
        ax=ax,
        colormap="viridis",
        colorbar=False,
    )
    ax.set_title(f"Sampling ratio = {multiplier}")
    sns.despine(ax=ax, offset=10)

plt.tight_layout()
plt.show()