import seaborn as sns
import pandas as pd
sns.set_theme()
# The seaborn namespace is flat; all of the functionality is accessible at the top level. 
# But the code itself is hierarchically structured, with modules of functions that achieve similar visualization goals through different means. 
# Most of the docs are structured around these modules: you’ll encounter names like “relational”, “distributional”, and “categorical”.

# For example, the distributions module defines functions that specialize in representing the distribution of datapoints. This includes familiar methods like the histogram:

penguins = sns.load_dataset("penguins")
print(sns.histplot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack"))

# Along with similar, but perhaps less familiar, options such as kernel density estimation:
print(sns.kdeplot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack"))

tips = sns.load_dataset("tips")

print(sns.relplot(
    data=tips,
    x="total_bill", y="tip", col="time",
    hue="smoker", style="smoker", size="size",
))

dots = sns.load_dataset("dots")
print(sns.relplot(
    data=dots, kind="line",
    x="time", y="firing_rate", col="align",
    hue="choice", size="coherence", style="choice",
    facet_kws=dict(sharex=False),
))