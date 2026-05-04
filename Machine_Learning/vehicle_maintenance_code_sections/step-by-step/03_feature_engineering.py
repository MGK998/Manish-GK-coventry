# ==============================================================================
# 03 Feature Engineering
# Curated file for screencast explanation.
# ==============================================================================


# ############################################################################
# Included section: 7.1 Average engineered-feature values by class
# ############################################################################

# ==============================================================================
# 7.1 Average engineered-feature values by class
# Extracted from: vehicle_maintenance_logreg_main.ipynb
# For presentation/reference. Run the original notebook for full execution.
# ==============================================================================

# --- Markdown cell 36 ---
# ### 7.1 Average engineered-feature values by class
# This helps show whether the engineered features separate vehicles that need maintenance from those that do not.

# --- Code cell 37 ---
engineered_group_means = engineered_df.groupby("Need_Maintenance")[engineered_only_cols].mean().T
display(engineered_group_means.round(2))
engineered_group_means.to_csv(OUTPUT_DIR / "engineered_feature_group_means.csv")

fig, ax = plt.subplots(figsize=(10, 5))
engineered_group_means.plot(kind="bar", ax=ax)
ax.set_title("Engineered feature means by target class")
ax.set_ylabel("Mean value")
ax.set_xticklabels(engineered_group_means.index, rotation=70, ha="right")
ax.legend(title="Need_Maintenance")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "engineered_feature_means_by_target.png", bbox_inches="tight")
plt.show()

