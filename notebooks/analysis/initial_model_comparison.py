"""
    notebooks\analysis\initial_model_comparison.py - a notebook to visualize the data and compare the results of different methods
"""

# %%
from pathlib import Path

import pandas as pd

_root = Path(__file__).parent.parent.parent  # root directory
res_dir = _root / "results"

assert res_dir.exists(), f"{res_dir} does not exist"

plot_out_dir = res_dir / "figures" / "model-analysis"
plot_out_dir.mkdir(exist_ok=True, parents=True)


# %%

filename = res_dir / "compiled_trained_model_performance.csv"

df = pd.read_csv(str(filename.resolve()))
df.info()


# %%

import plotly.express as px

height = 600
width = int(height * 1.618)

f1_df = df[df.performance_metric == "accuracy"]
fig_env = px.violin(f1_df, x="source", y="metric_value", color="source")

fig_env.show()

# %%

# bar plot of the 10 best models by balanced accuracy score
bas_df = df[df.performance_metric == "balanced_accuracy_score"]
bas_df.sort_values(by="metric_value", ascending=False, inplace=True)
bas_plot = bas_df.head(10)

fig_bas = px.bar(
    bas_plot, x="model_filename", y="metric_value", color="source", text="dataset"
)

fig_bas.show()


# %%

df_acc = df[df.performance_metric == "accuracy"]
fig_box = px.box(df_acc, x="dataset", y="metric_value", color="source")

# %%


df_mitbih = df[df.dataset == "mitbih"]
df_mitbih = df_mitbih[df_mitbih.performance_metric == "accuracy"]

# only keep top 50% of the models
df_mitbih = df_mitbih.sort_values(by="metric_value", ascending=False)
plot_mitbih = df_mitbih.head(int(len(df_mitbih) / 2))

mit_labels = {
    "model_filename": "Model",
    "accuracy": "Accuracy",
    "source": "Model Training Method",
    "dataset": "Dataset",
    "metric_value": "Accuracy Score",
}
mit_top = px.box(
    plot_mitbih,
    x="source",
    y="metric_value",
    color="source",
    template="presentation",
    labels=mit_labels,
    hover_data=["model_filename", "dataset", "source", "metric_value"],
    points="all",
    height=height,
    width=width,
    title="Mitbih - Accuracy",
)
mit_top.show()

mit_top.write_html(str(plot_out_dir / "mitbih_accuracy_comparison.html"))
# mit_top.write_image(str(plot_out_dir / "mitbih_accuracy_comparison.jpg"),  format="jpg", engine="kaleido")
# %%

df_ptbdb = df[df.dataset == "ptbdb"]
# filter for ROC AUC
df_ptbdb = df_ptbdb[df_ptbdb.performance_metric == "roc_auc_score"]
plot_ptbdb = df_ptbdb[df_ptbdb.metric_value >= 0.7]

ptb_labels = {
    "model_filename": "Model",
    "source": "Model Training Method",
    "dataset": "Dataset",
    "metric_value": "ROC AUC Score",
}

ptb_top = px.box(
    plot_ptbdb,
    x="source",
    y="metric_value",
    color="source",
    template="presentation",
    labels=ptb_labels,
    hover_data=["model_filename", "dataset", "source", "metric_value"],
    points="all",
    height=height,
    width=width,
    title="PTBDB - ROC AUC Score",
)
ptb_top.show()
# %%

ptb_top.write_html(str(plot_out_dir / "ptbdb_roc_auc_comparison.html"))
# ptb_top.write_image(str(plot_out_dir / "ptbdb_roc_auc_comparison.jpg"), format="jpg", engine="kaleido")
# %%
