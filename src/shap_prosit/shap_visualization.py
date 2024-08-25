import os
import sys
from operator import itemgetter
from pathlib import Path
from typing import Union

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from statistics import mean

import yaml
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.cluster import Birch, KMeans
from sklearn.decomposition import PCA

matplotlib.use("Agg")
MIN_OCCUR_AVG = 100
MIN_OCCUR_HEAT = 15


class ShapVisualization:
    def __init__(
        self,
        sv_path: Union[str, bytes, os.PathLike],
        ion: str,
        position_combos: list | None = None,
    ) -> None:
        if position_combos is None:
            self.position_combos = [
                [0, -1],
                [int(ion[1:].split("+")[0]) - 1, 0],
                [int(ion[1:].split("+")[0]) - 1, -1],
            ]
        else:
            self.position_combos = position_combos

        df = pd.read_parquet(sv_path)
        self.ion = ion

        # Get data from dataframe
        self.pred_intensities = df["intensity"].tolist()
        self.charge = df["charge"].tolist()
        self.energy = df["energy"].tolist()
        self.seq_list = df["sequence"].tolist()
        self.shap_values_list = df["shap_values"].tolist()

        # Initialize data structures
        self.count_positions = np.zeros((30))
        self.sv_sum = np.zeros((30))
        self.sv_abs_sum = np.zeros((30))
        self.combo_pos_sv_sum = []
        self.combo_pos_sv_abs_sum = []
        self.combo_inten = []
        self.amino_acids_sv = {}
        self.amino_acid_pos = {}
        self.amino_acid_pos_inten = {}

        for sequence, shap_values in zip(self.seq_list, self.shap_values_list):
            seq = np.array(sequence)
            sv = np.array(shap_values)

            le = len(seq)
            self.count_positions += np.append(np.ones((le)), np.zeros((30 - le)))

            # Gather sum and abs sum of SV in each position
            if self.ion[0] == "y":  # Reverse string for y-ion
                self.sv_sum[:le] += sv[::-1]
                self.sv_abs_sum[:le] += abs(sv[::-1])
            else:
                self.sv_sum[:le] += sv
                self.sv_abs_sum[:le] += abs(sv)

            self.__bi_token_combo(seq, sv)

            for i, (am_ac, sh_value) in enumerate(zip(seq, sv)):

                # Store SV for each AA
                if am_ac not in self.amino_acids_sv:
                    self.amino_acids_sv[am_ac] = []
                self.amino_acids_sv[am_ac].append(sh_value)

                # Create token for AA and position
                # 0 is the first index in ion, positives are inside ion
                # negatives are outside of ion
                tok_c = (
                    f"{am_ac}_{i + int(self.ion[1:].split('+')[0]) - le}"
                    if self.ion[0] == "y"
                    else f"{am_ac}_{int(self.ion[1:].split('+')[0]) - i - 1}"
                )

                # Check if token already in dictionary
                if tok_c not in self.amino_acid_pos:
                    self.amino_acid_pos[tok_c] = []
                if tok_c not in self.amino_acid_pos_inten:
                    self.amino_acid_pos_inten[tok_c] = []

                # Store values for token in list
                self.amino_acid_pos[tok_c].append(sh_value)
                self.amino_acid_pos_inten[tok_c].append(sum(sv))

        self.amino_acids_sorted = np.sort(list(self.amino_acids_sv.keys()))

        self.sv_avg = self.sv_sum / (self.count_positions + 1e-9)
        self.sv_avg *= self.count_positions > MIN_OCCUR_AVG
        self.sv_abs_avg = self.sv_abs_sum / (self.count_positions + 1e-9)
        self.sv_abs_avg *= self.count_positions > MIN_OCCUR_AVG

        self.amino_acids_abs_avg_sv = {
            a: np.mean(np.abs(self.amino_acids_sv[a])) for a in self.amino_acids_sorted
        }
        self.amino_acids_avg_sv = {
            a: np.mean(self.amino_acids_sv[a]) for a in self.amino_acids_sorted
        }
        self.amino_acids_std_sv = {
            a: np.std(self.amino_acids_sv[a]) for a in self.amino_acids_sorted
        }

        # AA-position heatmaps
        # Consolidate lists of values into single values for each token
        self.amino_acid_pos_inten = {
            tok: np.mean(self.amino_acid_pos_inten[tok])
            for tok in self.amino_acid_pos_inten.keys()
            if len(self.amino_acid_pos_inten[tok]) > MIN_OCCUR_HEAT
        }
        self.amino_acid_pos_avg = {
            tok: np.mean(self.amino_acid_pos[tok])
            for tok in self.amino_acid_pos.keys()
            if len(self.amino_acid_pos) > MIN_OCCUR_HEAT
        }
        self.amino_acid_pos_abs_avg = {
            tok: np.mean(np.abs(self.amino_acid_pos[tok]))
            for tok in self.amino_acid_pos.keys()
            if len(self.amino_acid_pos) > MIN_OCCUR_HEAT
        }

    def __bi_token_combo(self, sequence, shap_values):
        for i, combo in enumerate(self.position_combos):
            # Append new dictionary
            if i >= len(self.combo_pos_sv_sum):
                self.combo_pos_sv_sum.append({})
                self.combo_pos_sv_abs_sum.append({})
                self.combo_inten.append({})

            if self.ion[0] == "y":
                seq = sequence[::-1]
                sv = shap_values[::-1]
            else:
                seq = sequence
                sv = shap_values

            ion_size = int(self.ion[1:].split("+")[0])
            combo_pos = [-1 * combo[0] + ion_size - 1, -1 * combo[1] + ion_size - 1]

            if combo_pos[0] >= len(sequence) or combo_pos[1] >= len(sequence):
                continue

            # AA-AA token
            tok = f"{seq[combo_pos[0]]}-{seq[combo_pos[1]]}"

            # Initialize lists for new tokens
            if tok not in self.combo_pos_sv_sum[i]:
                self.combo_pos_sv_sum[i][tok] = []
            if tok not in self.combo_pos_sv_abs_sum[i]:
                self.combo_pos_sv_abs_sum[i][tok] = []
            if tok not in self.combo_inten[i]:
                self.combo_inten[i][tok] = []

            self.combo_pos_sv_sum[i][tok].append(sv[combo_pos[0]] + sv[combo_pos[1]])
            self.combo_pos_sv_abs_sum[i][tok].append(
                abs(sv[combo_pos[0]]) + abs(sv[combo_pos[1]])
            )
            self.combo_inten[i][tok].append(sum(sv))

    def clustering(self, config: dict):
        number_of_aas = config["clustering"]["number_of_aa"]
        if config["clustering"]["from_which_end"] == "left":
            cluster_sv = np.array(
                [shap_values[:number_of_aas] for shap_values in self.shap_values_list]
            )
        else:
            cluster_sv = np.array(
                [shap_values[-number_of_aas:] for shap_values in self.shap_values_list]
            )

        clustering = KMeans(n_clusters=3).fit_predict(cluster_sv)
        unique, counts = np.unique(clustering, return_counts=True)

        data = PCA().fit_transform(cluster_sv)
        plt.close("all")
        scatter = plt.scatter(
            data[:, 0], data[:, 1], marker=".", c=clustering, cmap="rainbow"
        )
        plt.legend(*scatter.legend_elements(), title="Clusters")
        plt.savefig(
            str(Path(config["sv_path"]).parent.absolute()) + "/clustering.png",
            bbox_inches="tight",
        )

        # Create folders for clusters
        for cluster in np.asarray((unique)):
            os.makedirs(
                str(Path(config["sv_path"]).parent.absolute()) + f"/cluster_{cluster}",
                exist_ok=True,
            )

        # Write info about clusters
        for cluster in np.asarray((unique)):
            with open(
                str(Path(config["sv_path"]).parent.absolute())
                + f"/cluster_{cluster}/info.txt",
                "w",
                encoding="utf-8",
            ) as f:
                mean_cluster_sv = np.mean(
                    np.array(self.pred_intensities)[np.where(clustering == cluster)[0]]
                )
                cluster_len = len(
                    np.array(self.pred_intensities)[np.where(clustering == cluster)[0]]
                )
                f.write(
                    f"Length of the cluster: {cluster_len}, Cluster mean intensity: {mean_cluster_sv}\n"
                )

        # Write sequences and shap values in files
        for cluster in np.asarray((unique)):
            output = {
                "sequence": list(
                    itemgetter(*np.where(clustering == cluster)[0])(self.seq_list)
                ),
                "shap_values": list(
                    itemgetter(*np.where(clustering == cluster)[0])(
                        self.shap_values_list
                    )
                ),
                "intensity": list(
                    np.array(self.pred_intensities)[np.where(clustering == cluster)[0]]
                ),
                "energy": list(
                    np.array(self.energy)[np.where(clustering == cluster)[0]]
                ),
                "charge": list(
                    np.array(self.charge)[np.where(clustering == cluster)[0]]
                ),
            }
            pd.DataFrame(output).to_parquet(
                str(Path(config["sv_path"]).parent.absolute())
                + f"/cluster_{cluster}/output.parquet.gzip",
                compression="gzip",
            )

        # Create plots for output.txt in clusters
        for cluster in np.asarray((unique)):
            visualization = ShapVisualization(
                str(Path(config["sv_path"]).parent.absolute())
                + f"/cluster_{cluster}/output.parquet.gzip",
                self.ion,
                [
                    [0, number_of_aas - 1],
                    [1, number_of_aas - 2],
                    [0, 1],
                    [number_of_aas - 2, number_of_aas - 1],
                ],
            )
            visualization.full_report(
                save=str(Path(config["sv_path"]).parent.absolute())
                + f"/cluster_{cluster}"
            )

    def aa_only_plot(self, save=False):
        plt.close("all")
        fig, axes = plt.subplots(4, 1)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        axes[0].set_title("mean(abs(sv))")
        axes[0].set_xlim([-0.5, 19.5])
        axes[0].plot(list(self.amino_acids_abs_avg_sv.values()), "ro")
        im = axes[1].imshow(
            np.array(list(self.amino_acids_abs_avg_sv.values()))[None],
            cmap="RdBu_r",
            norm=TwoSlopeNorm(0),
        )
        axes[2].set_title("mean(sv)")
        axes[2].errorbar(
            np.arange(len(self.amino_acids_sorted)),
            np.array(list(self.amino_acids_avg_sv.values())),
            np.array(list(self.amino_acids_std_sv.values())),
            marker="o",
            linestyle="none",
            markerfacecolor="red",
            markersize=10,
        )
        im2 = axes[3].imshow(
            np.array(list(self.amino_acids_avg_sv.values()))[None],
            cmap="RdBu_r",
            norm=TwoSlopeNorm(0),
        )
        axes[3].set_xlim([-0.5, 19.5])
        for ax in axes:
            ax.set_xticks(np.arange(len(self.amino_acids_sorted)))
            ax.set_xticklabels(self.amino_acids_sorted, size=8)
        for ax in axes[[1, 3]]:
            ax.set_yticks([])
            ax.set_yticklabels([])
        fig.colorbar(im, ax=axes[:2]).ax.set_yscale("linear")
        fig.colorbar(im2, ax=axes[2:]).ax.set_yscale("linear")
        if save is not False:
            plt.savefig(save + "/aa_only_plot.png", bbox_inches="tight")
        else:
            plt.show()

    def position_only_plot(self, save=False):
        plt.close("all")
        fig, axes = plt.subplots(2)
        fig.set_figheight(5)
        fig.set_figwidth(15)
        axes[0].set_title("mean(abs(sv))")
        im = axes[0].imshow(self.sv_abs_avg[None], cmap="RdBu_r", norm=TwoSlopeNorm(0))
        axes[1].set_title("mean(sv)")
        im2 = axes[1].imshow(self.sv_avg[None], cmap="RdBu_r", norm=TwoSlopeNorm(0))
        tick_range = np.arange(
            int(self.ion[1:].split("+")[0]) - 1,
            -1 * (30 - int(self.ion[1:].split("+")[0]) + 1),
            -1,
        )
        for ax in axes:
            ax.axvline(
                x=int(self.ion[1:].split("+")[0]) - 0.5, color="black", linewidth=3
            )
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks(np.arange(30))
            ax.set_xticklabels(tick_range, size=6)
        fig.colorbar(im).ax.set_yscale("linear")
        fig.colorbar(im2).ax.set_yscale("linear")
        if save is not False:
            plt.savefig(save + "/position_only_plot.png", bbox_inches="tight")
        else:
            plt.show()

    def position_heatmap(self, save=False):
        plt.close("all")
        heatmap_int = np.zeros((len(self.amino_acids_sorted), 30))
        heatmap = np.zeros((len(self.amino_acids_sorted), 30))
        heatmap_abs = np.zeros((len(self.amino_acids_sorted), 30))

        for A, a in enumerate(self.amino_acids_sorted):
            for b in np.arange(30):
                tok = "%c_%d" % (a, -1 * (b + 1 - int(self.ion[1:].split("+")[0])))
                if tok in self.amino_acid_pos_inten:
                    heatmap_int[A, b] = self.amino_acid_pos_inten[tok]
                if tok in self.amino_acid_pos_abs_avg:
                    heatmap[A, b] = self.amino_acid_pos_avg[tok]
                    heatmap_abs[A, b] = self.amino_acid_pos_abs_avg[tok]

        fig, axes = plt.subplots(3)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        axes[0].set_title("mean(intensity)")
        im = axes[0].imshow(heatmap_int, cmap="RdBu_r", norm=TwoSlopeNorm(0))
        axes[1].set_title("mean(abs(sv))")
        im2 = axes[1].imshow(heatmap_abs, cmap="RdBu_r", norm=TwoSlopeNorm(0))
        axes[2].set_title("mean(sv)")
        im3 = axes[2].imshow(heatmap, cmap="RdBu_r", norm=TwoSlopeNorm(0))

        tick_range = np.arange(
            int(self.ion[1:].split("+")[0]) - 1,
            -1 * (30 - int(self.ion[1:].split("+")[0]) + 1),
            -1,
        )
        for ax in axes:
            ax.axvline(
                x=int(self.ion[1:].split("+")[0]) - 0.5, color="black", linewidth=3
            )
            ax.set_yticks(np.arange(len(self.amino_acids_sorted)))
            ax.set_yticklabels(self.amino_acids_sorted, size=6)
            ax.set_xticks(np.arange(30))
            ax.set_xticklabels(tick_range, size=6)
        fig.colorbar(im).ax.set_yscale("linear")
        fig.colorbar(im2).ax.set_yscale("linear")
        fig.colorbar(im3).ax.set_yscale("linear")
        if save is not False:
            plt.savefig(save + "/position_heatmap.png", bbox_inches="tight")
        else:
            plt.show()

    def aa_heatmap(self, save=False):
        plt.close("all")
        fig, axes = plt.subplots(3, len(self.position_combos))
        fig.set_figheight(15)
        fig.set_figwidth(17)

        for ax in axes.flatten():
            ax.set_yticks(np.arange(len(self.amino_acids_sorted)))
            ax.set_yticklabels(self.amino_acids_sorted, size=6)
            ax.set_ylabel("AA(L)")
            ax.set_xticks(np.arange(len(self.amino_acids_sorted)))
            ax.set_xticklabels(self.amino_acids_sorted, size=6)
            ax.set_xlabel("AA(R)")

        for i, combo in enumerate(self.position_combos):
            heatmap = np.zeros(
                (len(self.amino_acids_sorted), len(self.amino_acids_sorted))
            )
            heatmap_abs = np.zeros(
                (len(self.amino_acids_sorted), len(self.amino_acids_sorted))
            )
            heatmap_int = np.zeros(
                (len(self.amino_acids_sorted), len(self.amino_acids_sorted))
            )
            for l, aa1 in enumerate(self.amino_acids_sorted):
                for m, aa2 in enumerate(self.amino_acids_sorted):
                    tok = "%c-%c" % (aa1, aa2)
                    if tok in self.combo_pos_sv_sum[i]:
                        if len(self.combo_pos_sv_sum[i][tok]) > MIN_OCCUR_HEAT:
                            heatmap[l, m] = np.mean(self.combo_pos_sv_sum[i][tok])
                            heatmap_abs[l, m] = np.mean(
                                self.combo_pos_sv_abs_sum[i][tok]
                            )
                            heatmap_int[l, m] = np.mean(self.combo_inten[i][tok])

            axes[0, i].set_title("%d_%d: mean(intensity)" % tuple(combo))
            im = axes[0, i].imshow(heatmap_int, cmap="RdBu_r", norm=TwoSlopeNorm(0))
            axes[1, i].set_title("%d_%d: mean(sv)" % tuple(combo))
            im2 = axes[1, i].imshow(heatmap, cmap="RdBu_r", norm=TwoSlopeNorm(0))
            axes[2, i].set_title("mean(abs(%d)+abs(%d))" % tuple(combo))
            im3 = axes[2, i].imshow(heatmap_abs, cmap="RdBu_r", norm=TwoSlopeNorm(0))

        fig.colorbar(im, shrink=0.7).ax.set_yscale("linear")
        fig.colorbar(im2, shrink=0.7).ax.set_yscale("linear")
        fig.colorbar(im3, shrink=0.7).ax.set_yscale("linear")

        if save is not False:
            plt.savefig(save + "/aa_heatmap.png", bbox_inches="tight")
        else:
            plt.show()

    def swarmplot(self, save=False):
        plt.close("all")
        fig, ax = plt.subplots()
        fig.set_figwidth(15)
        fig.set_figheight(8)

        data = {"shap_value": [], "amino_acid": [], "inside_ion": []}
        for key, values in self.amino_acid_pos.items():
            inside = True if int(key[2:]) >= 0 else False
            data["shap_value"].extend(values)
            data["amino_acid"].extend([key[0]] * len(values))
            data["inside_ion"].extend([inside] * len(values))

        plot = sns.stripplot(
            data=data,
            order=self.amino_acids_sorted,
            x="shap_value",
            y="amino_acid",
            size=2,
            jitter=0.4,
            hue="inside_ion",
            legend=True,
            ax=ax,
        )
        plt.axvline(x=0, color="black", linewidth=4)

        # cmap = plt.get_cmap("RdBu_r")
        # norm = plt.Normalize(min(data["position"]), max(data["position"]))
        # sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        # sm.set_array([])
        # cbar = fig.colorbar(sm, ax=plot, shrink=0.7)
        # cbar.set_label("Position")

        if save is not False:
            plt.savefig(save + "/swarmplot.png")
        else:
            plt.show()

    def boxplot_position(self, save="."):
        plt.close("all")
        fig = plt.gcf()
        fig.set_figwidth(15)

        sum_abs_sv = {}

        for key in self.amino_acid_pos.keys():
            if key.startswith(("R", "H", "K")):
                continue
            if len(self.amino_acid_pos[key]) < MIN_OCCUR_HEAT:
                continue
            sum_abs_sv[key] = mean([abs(x) for x in self.amino_acid_pos[key]])

        data = {"SHAP values": [], "Amino acid - Position": []}

        for aa in list(
            dict(
                sorted(sum_abs_sv.items(), key=lambda x: x[1], reverse=True)[:20]
            ).keys()
        ):
            for shap in self.amino_acid_pos[aa]:
                data["SHAP values"].append(abs(shap))
                data["Amino acid - Position"].append(aa)

        df = pd.DataFrame(data)
        figure = sns.boxplot(
            data=df,
            x="Amino acid - Position",
            y="SHAP values",
            color="#1f77b4",
        ).set_title("mean(abs(sv))")

        if save is not False:
            plt.savefig(save + "/boxplot_position.png", bbox_inches="tight")
        else:
            plt.show()

    def boxplot_token(self, save="."):
        plt.close("all")
        fig = plt.gcf()
        fig.set_figwidth(15)

        sum_abs_sv = {}

        for key in self.combo_pos_sv_sum[0].keys():
            if len(self.combo_pos_sv_sum[0][key]) < MIN_OCCUR_HEAT:
                continue
            sum_abs_sv[key] = mean([abs(x) for x in self.combo_pos_sv_sum[0][key]])

        data = {"SHAP values": [], "Amino acids on positions 0:-1": []}

        for aa in list(
            dict(
                sorted(sum_abs_sv.items(), key=lambda x: x[1], reverse=True)[:20]
            ).keys()
        ):
            for shap in self.combo_pos_sv_sum[0][aa]:
                data["SHAP values"].append(abs(shap))
                data["Amino acids on positions 0:-1"].append(aa)

        df = pd.DataFrame(data)
        figure = sns.boxplot(
            data=df,
            x="Amino acids on positions 0:-1",
            y="SHAP values",
            color="#1f77b4",
        ).set_title("mean(abs(sv))")

        if save is not False:
            plt.savefig(save + "/boxplot_token.png", bbox_inches="tight")
        else:
            plt.show()

    def full_report(self, save="."):
        self.aa_only_plot(save=save)
        self.position_only_plot(save=save)
        self.position_heatmap(save=save)
        self.aa_heatmap(save=save)
        self.swarmplot(save=save)
        self.boxplot_position(save=save)
        self.boxplot_token(save=save)


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as file:
        config = yaml.safe_load(file)
    visualization = ShapVisualization(
        config["shap_visualization"]["sv_path"], ion=config["shap_calculator"]["ion"]
    )
    visualization.full_report(
        save=str(Path(config["shap_visualization"]["sv_path"]).parent.absolute())
    )
    visualization.clustering(config["shap_visualization"])
