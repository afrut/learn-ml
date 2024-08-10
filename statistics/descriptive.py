import os
import pickle as pk

import dfutl
import plots

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(os.path.join(script_dir, "../data/iris.pkl"))
    output_path = os.path.abspath(os.path.join(script_dir, "../visualization/outputs"))

    # Load some data.
    with open(file_path, "rb") as fl:
        df = pk.load(fl)
        cols = dfutl.numericColumns(df)
        df = df.loc[:, cols]

    # Numerical summaries of data
    print(df.describe())

    plots.stemleaf(
        df,
        title="Stem and Leaf",
        save=True,
        savepath=f"{output_path}/iris_stemleaf.txt",
    )

    plots.histogram(df, save=True, save_path=f"{output_path}/iris_histogram.png")

    plots.boxplot(df, save=True, savepath=f"{output_path}/iris_boxplot.png")

    plots.scattermatrix(df, save=True, savepath=f"{output_path}/iris_scattermatrix.png")

    plots.heatmap(df, save=True, savepath=f"{output_path}/iris_heatmap.png")

    plots.probplot(df, save=True, savepath=f"{output_path}/iris_probplot.png")
