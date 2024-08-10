# exec(open('visualization.py').read())
import pickle as pk

import datacfg
import plots

if __name__ == "__main__":
    datacfg = datacfg.datacfg()
    for datasetname in datacfg.keys():
        print(datasetname)
        df = pk.load(open(datacfg[datasetname]["filepath"], "rb"))

        plots.stemleaf(
            df,
            title="Stem and Leaf",
            save=True,
            savepath="./visualization/outputs/plots/" + datasetname + "_stemleaf.txt",
        )

        plots.histogram(
            df,
            save=True,
            save_path="./visualization/outputs/plots/" + datasetname + "_histogram.png",
        )

        plots.boxplot(
            df,
            save=True,
            savepath="./visualization/outputs/plots/" + datasetname + "_boxplot.png",
            close=True,
        )

        plots.scattermatrix(
            df,
            save=True,
            savepath="./visualization/outputs/plots/"
            + datasetname
            + "_scattermatrix.png",
            close=True,
        )

        plots.heatmap(
            df,
            save=True,
            savepath="./visualization/outputs/plots/" + datasetname + "_heatmap.png",
            close=True,
        )

        plots.probplot(
            df,
            save=True,
            savepath="./visualization/outputs/plots/" + datasetname + "_probplot.png",
            close=True,
        )
