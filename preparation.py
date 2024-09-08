import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def get_dataset(
    df: pd.DataFrame,
    target_name: str
) -> (pd.DataFrame, pd.Series, list[str], list[str]):
    """
    Splits a DataFrame into features and target, and identifies numeric and categorical features.

    Parameters:
    - df: pd.DataFrame
        The input DataFrame containing the dataset.
    - target_name: str
        The name of the column in the DataFrame to be used as the target variable.

    Returns:
    - X: pd.DataFrame
        DataFrame containing the feature columns.
    - y: pd.Series
        Series containing the target variable.
    - num_features: list[str]
        List of names of numeric features.
    - cat_features: list[str]
        List of names of categorical features.
    """
    X, y = df.drop(target_name, axis=1), df[target_name]
    
    # Identify numeric features
    num_features = X.columns[pd.DataFrame(X).apply(pd.api.types.is_numeric_dtype)].tolist()
    
    # Identify categorical features
    cat_features = [c for c in X.columns if c not in num_features]
    
    return X, y, num_features, cat_features

def train_test_split_fit_predict(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    models: dict,
    test_size: float = 0.3,
    random_state: int = 42,
    stratify=None,
) -> (dict, dict):
    """
    Splits the dataset into training and testing sets, fits the models on the training set, 
    and makes predictions on both training and testing sets.

    Parameters:
    - X_full: pd.DataFrame
        The full DataFrame of features.
    - y_full: pd.Series
        The full Series of target values.
    - models: dict
        Dictionary where keys are model names and values are model instances.
    - test_size: float, optional (default=0.3)
        The proportion of the dataset to include in the test split.
    - random_state: int, optional (default=42)
        The seed used by the random number generator.
    - stratify: array-like, optional (default=None)
        If not None, data is split in a stratified fashion, using this as the class labels.

    Returns:
    - trained_models: dict
        Dictionary where keys are model names and values are trained model instances.
    - predictions: dict
        Dictionary with keys 'train' and 'test', and values being dictionaries of model names to predictions.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    # Fit models on the training set
    trained_models = {
        model_name: model.fit(X_train, y_train)
        for model_name, model in models.items()
    }

    # Make predictions on training and testing sets
    predictions = {
        'train': {
            model_name: pd.Series(model.predict(X_train), index=y_train.index)
            for model_name, model in trained_models.items()
        },
        'test': {
            model_name: pd.Series(model.predict(X_test), index=y_test.index)
            for model_name, model in trained_models.items()
        }
    }

    return trained_models, predictions

def calculate_metrics(
    y: dict,
    pred: dict,
    models: dict
) -> pd.DataFrame:
    """
    Calculates performance metrics (MAE and R^2) for each model on training and testing datasets.

    Parameters:
    - y: dict
        Dictionary where keys are 'train' and 'test', and values are Series of true target values.
    - pred: dict
        Dictionary with keys 'train' and 'test', and values being dictionaries of model names to predictions.
    - models: dict
        Dictionary of model names to model instances.

    Returns:
    - results_format: pd.DataFrame
        DataFrame with calculated metrics formatted as strings. Columns include MAE and R^2 for each model.
    """
    results = pd.DataFrame()

    # Compute metrics for each sample, model, and metric type
    for sample in y.keys():
        for metric_name, metric in zip(["MAE", "R^2"], [mean_absolute_error, r2_score]):
            for model_name in models.keys():
                # Compute the metric and store in DataFrame
                results.loc[sample, f"{metric_name}({model_name})"] = metric(y_true=y[sample], y_pred=pred[sample][model_name])

    # Format the results DataFrame
    results_format = results.apply(lambda col:
        col.apply(lambda x: f"{x:.1%}" if "R^2(" in col.name else (f"$ {x:,.0f}" if "MAE(" in col.name else x))
    )
    
    return results_format

def plot_decision_trees(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    model_names: list[str],
    output_file: str = "decision_tree_heatmap.png"
):
    """
    Plots and compares decision trees for given models.

    Parameters:
    - models: dict
        Dictionary where keys are model names and values are trained decision tree models.
    - X: pd.DataFrame
        DataFrame of feature values used to fit the models.
    - y: pd.Series
        Series of target values used to fit the models.
    - model_names: list[str]
        List of model names to be plotted.
    - output_file: str, optional (default="decision_tree_heatmap.png")
        File name to save the plot.
    
    Raises:
    - ValueError: if the DataFrame does not have at least two columns.
    """
    if len(X.columns) < 2:
        raise ValueError("The input DataFrame must have at least two columns for plotting.")
    
    xlim_global = (X[X.columns[0]].min(), X[X.columns[0]].max())
    ylim_global = (X[X.columns[1]].min(), X[X.columns[1]].max())
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.10)

    for en_ax, model_name in enumerate(model_names):
        ax = axs[en_ax]
        tree = models[model_name].tree_

        # Initialize xlim and ylim arrays
        xlim = np.full((tree.node_count, 2), xlim_global)
        ylim = np.full((tree.node_count, 2), ylim_global)

        for en, (f, t, cl, cr) in enumerate(zip(tree.feature, tree.threshold, tree.children_left, tree.children_right)):
            if cl != -1:  # Check if left child exists
                xlim[cl] = xlim[en]; ylim[cl] = ylim[en]
            if cr != -1:  # Check if right child exists
                xlim[cr] = xlim[en]; ylim[cr] = ylim[en]

            if f == 0:  # Vertical split
                xlim[cl, 1] = t; xlim[cr, 0] = t
                ax.plot([t] * 2, ylim[en], lw=0.5, color="grey")
            elif f == 1:  # Horizontal split
                ylim[cr, 0] = t; ylim[cl, 1] = t
                ax.plot(xlim[en], [t] * 2, lw=0.5, color="grey")
            else:  # Leaf node
                color_value = tree.value.reshape(-1)[en] / y.max()
                ax.fill_between(xlim[en][:, 0], ylim[en][:, 0], ylim[en][:, 1], color=plt.cm.Reds(color_value), zorder=-1)
                value = round(tree.value.reshape(-1)[en] / 1000)
                ycoords = np.mean(ylim[en]) + (.7 if value in (177, 247) else -0.7 if value in (157, 205, 283) else 0)
                ax.annotate(f"$\n{value}\nk", [np.mean(xlim[en]), ycoords], ha="center", va="center")

        ax.set_xticks(np.arange(1_000, 7_000, 1_000))
        ax.set_xticklabels([f"{x:,.0f}" for x in np.arange(1_000, 7_000, 1_000)])
        ax.set_xlim(xlim_global)
        ax.set_ylim(ylim_global)
        ax.set_xlabel(X.columns[0])
        if en_ax == 0:
            ax.set_ylabel(X.columns[1])
        ax.set_title(f"Predicted {y.name}\n({model_name})", fontsize=10)

    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.show()
