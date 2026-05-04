# ==============================================================================
# 09 Results Metrics Curves Confusion
# Curated file for screencast explanation.
# ==============================================================================


# ############################################################################
# Included section: 10) Metrics comparison
# ############################################################################

# ==============================================================================
# 10) Metrics comparison
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 50 ---
# ## 10) Metrics comparison
# These are the main numeric results you usually show in the final report.

# --- Code cell 51 ---
plot_metrics = model_summary.set_index("model")[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]

fig, ax = plt.subplots(figsize=(10, 5.5))
plot_metrics.plot(kind="bar", ax=ax)
ax.set_ylim(0, 1.05)
ax.set_title("Logistic Regression metric comparison")
ax.set_ylabel("Score")
ax.set_xticklabels(plot_metrics.index, rotation=0)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "metric_comparison.png", bbox_inches="tight")
plt.show()



# ############################################################################
# Included section: 11) ROC curves
# ############################################################################

# ==============================================================================
# 11) ROC curves
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 52 ---
# ## 11) ROC curves

# --- Code cell 53 ---
fig, ax = plt.subplots(figsize=(7, 5))
for name, res in results.items():
    ax.plot(res["fpr"], res["tpr"], label=f"{name} (AUC={res['tuned_metrics']['roc_auc']:.3f})")

ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curves")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "roc_curves.png", bbox_inches="tight")
plt.show()



# ############################################################################
# Included section: 12) Precision-recall curves
# ############################################################################

# ==============================================================================
# 12) Precision-recall curves
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 54 ---
# ## 12) Precision-recall curves

# --- Code cell 55 ---
fig, ax = plt.subplots(figsize=(7, 5))
for name, res in results.items():
    ax.plot(res["pr_recall"], res["pr_precision"], label=f"{name} (PR-AUC={res['tuned_metrics']['pr_auc']:.3f})")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall curves")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "precision_recall_curves.png", bbox_inches="tight")
plt.show()



# ############################################################################
# Included section: 13) Tuned-threshold confusion matrices
# ############################################################################

# ==============================================================================
# 13) Tuned-threshold confusion matrices
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 56 ---
# ## 13) Tuned-threshold confusion matrices

# --- Code cell 57 ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, (name, res) in zip(axes, results.items()):
    m = res["tuned_metrics"]
    matrix = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    im = ax.imshow(matrix)
    ax.set_title(name)
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:,}", ha="center", va="center")

fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.04)
fig.suptitle("Confusion matrices at tuned thresholds")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "confusion_matrices.png", bbox_inches="tight")
plt.show()

