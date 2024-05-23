import os
import sys
from pathlib import Path
from typing import Union

import numpy as np
import seaborn as sns
import yaml
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.cluster import Birch, KMeans
from sklearn.decomposition import PCA

MIN_OCCUR_AVG = 100
MIN_OCCUR_HEAT = 5


class ShapVisualization:
    def __init__(
        self, sv_path: Union[str, bytes, os.PathLike], position_combos: list = None
    ) -> None:
        if position_combos is None:
            self.position_combos = [[1, 9], [8, 9], [9, 10], [1, 10]]
        else:
            self.position_combos = position_combos

        with open(sv_path) as f:
            lines = np.array(f.read().split("\n"))[1:-1]

        sequences = list(lines[::2])
        svs = list(lines[1::2])

        self.pred_intensities = [float(line.split()[-1]) for line in sequences]
        self.charge = [int(line.split()[-2]) for line in sequences]
        self.energy = [float(line.split()[-3]) for line in sequences]
        self.seq_list = [line.split()[:-3] for line in sequences]

        self.shap_values_list = [[float(m) for m in line.split()] for line in svs]

        self.count_positions = np.zeros((30))
        self.sv_sum_from_left = np.zeros((30))
        self.sv_sum_from_right = np.zeros((30))

        self.sv_abs_sum_from_left = np.zeros((30))
        self.sv_abs_sum_from_right = np.zeros((30))

        self.combo_pos_sv_sum = []
        self.combo_pos_sv_abs_sum = []
        self.combo_sv_sum = []

        self.amino_acids_sv = {}
        self.amino_acid_pos_from_left = {}
        self.amino_acid_pos_from_right = {}
        self.amino_acid_pos_sv_sum_from_left = {}
        self.amino_acid_pos_sv_sum_from_right = {}

        for sequence, shap_values in zip(self.seq_list, self.shap_values_list):
            seq = np.array(sequence)
            sv = np.array(shap_values)

            le = len(seq)
            self.count_positions += np.append(np.ones((le)), np.zeros((30 - le)))
            self.sv_sum_from_left[:le] += sv
            self.sv_sum_from_right[:le] += sv[::-1]

            self.sv_abs_sum_from_left[:le] += abs(sv)
            self.sv_abs_sum_from_right[:le] += abs(sv[::-1])

            self.__bi_token_combo(seq, sv)

            for i, (am_ac, sh_value) in enumerate(zip(seq, sv)):

                # Store amino acid sv in list
                if am_ac not in self.amino_acids_sv:
                    self.amino_acids_sv[am_ac] = []
                self.amino_acids_sv[am_ac].append(sh_value)

                # Define token as position from left end
                tok_le = f"{am_ac}-{i}"
                if tok_le not in self.amino_acid_pos_from_left:
                    self.amino_acid_pos_from_left[tok_le] = []
                if tok_le not in self.amino_acid_pos_sv_sum_from_left:
                    self.amino_acid_pos_sv_sum_from_left[tok_le] = []
                # Store values for token in list
                self.amino_acid_pos_from_left[tok_le].append(sh_value)
                self.amino_acid_pos_sv_sum_from_left[tok_le].append(sum(sv))

                # Define token as position from right end
                tok_re = f"{am_ac}-{le - i - 1}"
                if tok_re not in self.amino_acid_pos_from_right:
                    self.amino_acid_pos_from_right[tok_re] = []
                if tok_re not in self.amino_acid_pos_sv_sum_from_right:
                    self.amino_acid_pos_sv_sum_from_right[tok_re] = []
                # Store values for token in list
                self.amino_acid_pos_from_right[tok_re].append(sh_value)
                self.amino_acid_pos_sv_sum_from_right[tok_re].append(sum(sv))

        self.amino_acids_sorted = np.sort(list(self.amino_acids_sv.keys()))

        self.sv_avg_from_left = self.sv_sum_from_left / (self.count_positions + 1e-9)
        self.sv_avg_from_left *= self.count_positions > MIN_OCCUR_AVG
        self.sv_avg_from_right = self.sv_sum_from_right / (self.count_positions + 1e-9)
        self.sv_avg_from_right *= self.count_positions > MIN_OCCUR_AVG
        self.sv_abs_avg_from_left = self.sv_abs_sum_from_left / (
            self.count_positions + 1e-9
        )
        self.sv_abs_avg_from_left *= self.count_positions > MIN_OCCUR_AVG
        self.sv_abs_avg_from_right = self.sv_abs_sum_from_right / (
            self.count_positions + 1e-9
        )
        self.sv_abs_avg_from_right *= self.count_positions > MIN_OCCUR_AVG

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
        self.amino_acid_pos_sv_sum_from_left = {
            tok: np.mean(self.amino_acid_pos_sv_sum_from_left[tok])
            for tok in self.amino_acid_pos_sv_sum_from_left.keys()
            if len(self.amino_acid_pos_sv_sum_from_left[tok]) > MIN_OCCUR_HEAT
        }
        self.amino_acid_pos_avg_from_left = {
            tok: np.mean(self.amino_acid_pos_from_left[tok])
            for tok in self.amino_acid_pos_from_left.keys()
            if len(self.amino_acid_pos_from_left) > MIN_OCCUR_HEAT
        }
        self.amino_acid_pos_abs_avg_from_left = {
            tok: np.mean(np.abs(self.amino_acid_pos_from_left[tok]))
            for tok in self.amino_acid_pos_from_left.keys()
            if len(self.amino_acid_pos_from_left) > MIN_OCCUR_HEAT
        }
        self.amino_acid_pos_sv_sum_from_right = {
            tok: np.mean(self.amino_acid_pos_sv_sum_from_right[tok])
            for tok in self.amino_acid_pos_sv_sum_from_right.keys()
            if len(self.amino_acid_pos_sv_sum_from_right[tok]) > MIN_OCCUR_HEAT
        }
        self.amino_acid_pos_avg_from_right = {
            tok: np.mean(self.amino_acid_pos_from_right[tok])
            for tok in self.amino_acid_pos_from_right.keys()
            if len(self.amino_acid_pos_from_right[tok]) > MIN_OCCUR_HEAT
        }
        self.amino_acid_pos_abs_avg_from_right = {
            tok: np.mean(np.abs(self.amino_acid_pos_from_right[tok]))
            for tok in self.amino_acid_pos_from_right.keys()
            if len(self.amino_acid_pos_from_right[tok]) > MIN_OCCUR_HEAT
        }

    def __bi_token_combo(self, sequence, shap_values):
        for i, combo in enumerate(self.position_combos):
            # Add new dictionary, if there more combos
            if i >= len(self.combo_pos_sv_sum):
                self.combo_pos_sv_sum.append({})
                self.combo_pos_sv_abs_sum.append({})
                self.combo_sv_sum.append({})

            if combo[0] >= len(sequence) or combo[1] >= len(sequence):
                continue

            tok = f"{sequence[-combo[0]]}-{sequence[-combo[1]]}"

            # Initialize lists for new tokens
            if tok not in self.combo_pos_sv_sum[i]:
                self.combo_pos_sv_sum[i][tok] = []
            if tok not in self.combo_pos_sv_abs_sum[i]:
                self.combo_pos_sv_abs_sum[i][tok] = []
            if tok not in self.combo_sv_sum[i]:
                self.combo_sv_sum[i][tok] = []

            self.combo_pos_sv_sum[i][tok].append(
                shap_values[-combo[0]] + shap_values[-combo[1]]
            )
            self.combo_pos_sv_abs_sum[i][tok].append(
                abs(shap_values[-combo[0]]) + abs(shap_values[-combo[1]])
            )
            self.combo_sv_sum[i][tok].append(sum(shap_values))

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

        # Write header for clusters' output.txt
        for cluster in np.asarray((unique)):
            with open(
                str(Path(config["sv_path"]).parent.absolute())
                + f"/cluster_{cluster}/output.txt",
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
        for i in range(len(cluster_sv)):
            with open(
                str(Path(config["sv_path"]).parent.absolute())
                + f"/cluster_{clustering[i]}/output.txt",
                "a",
                encoding="utf-8",
            ) as f:
                f.write(
                    " ".join(self.seq_list[i])
                    + f" {self.energy[i]:.2f} {self.charge[i]} {self.pred_intensities[i]}\n"
                )
                f.write(" ".join(["%s" % m for m in self.shap_values_list[i]]) + "\n")

        # Create plots for output.txt in clusters
        for cluster in np.asarray((unique)):
            visualization = ShapVisualization(
                str(Path(config["sv_path"]).parent.absolute())
                + f"/cluster_{cluster}/output.txt",
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
        fig, axes = plt.subplots(4)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        axes[0].set_title("mean(abs(sv)) from left end")
        im = axes[0].imshow(
            self.sv_abs_avg_from_left[None], cmap="RdBu_r", norm=TwoSlopeNorm(0)
        )
        axes[1].set_title("mean(sv) from left end")
        im2 = axes[1].imshow(
            self.sv_avg_from_left[None], cmap="RdBu_r", norm=TwoSlopeNorm(0)
        )
        axes[2].set_title("mean(abs(sv)) from right end")
        im3 = axes[2].imshow(
            self.sv_abs_avg_from_right[None], cmap="RdBu_r", norm=TwoSlopeNorm(0)
        )
        axes[3].set_title("mean(sv) from right end")
        im4 = axes[3].imshow(
            self.sv_avg_from_right[None], cmap="RdBu_r", norm=TwoSlopeNorm(0)
        )
        for ax in axes:
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks(np.arange(30))
            ax.set_xticklabels(np.arange(30), size=6)
        fig.colorbar(im).ax.set_yscale("linear")
        fig.colorbar(im2).ax.set_yscale("linear")
        fig.colorbar(im3).ax.set_yscale("linear")
        fig.colorbar(im4).ax.set_yscale("linear")
        if save is not False:
            plt.savefig(save + "/position_only_plot.png", bbox_inches="tight")
        else:
            plt.show()

    def position_heatmap(self, save=False):
        plt.close("all")
        heatmap_int_le = np.zeros((len(self.amino_acids_sorted), 30))
        heatmap_le = np.zeros((len(self.amino_acids_sorted), 30))
        heatmap_abs_le = np.zeros((len(self.amino_acids_sorted), 30))
        heatmap_int_re = np.zeros((len(self.amino_acids_sorted), 30))
        heatmap_re = np.zeros((len(self.amino_acids_sorted), 30))
        heatmap_abs_re = np.zeros((len(self.amino_acids_sorted), 30))

        for A, a in enumerate(self.amino_acids_sorted):
            for b in np.arange(30):
                tok = "%c-%d" % (a, b)
                if tok in self.amino_acid_pos_sv_sum_from_left:
                    heatmap_int_le[A, b] = self.amino_acid_pos_sv_sum_from_left[tok]
                if tok in self.amino_acid_pos_abs_avg_from_left:
                    heatmap_le[A, b] = self.amino_acid_pos_avg_from_left[tok]
                    heatmap_abs_le[A, b] = self.amino_acid_pos_abs_avg_from_left[tok]
                if tok in self.amino_acid_pos_sv_sum_from_right:
                    heatmap_int_re[A, b] = self.amino_acid_pos_sv_sum_from_right[tok]
                if tok in self.amino_acid_pos_abs_avg_from_right:
                    heatmap_re[A, b] = self.amino_acid_pos_avg_from_right[tok]
                    heatmap_abs_re[A, b] = self.amino_acid_pos_abs_avg_from_right[tok]

            fig, axes = plt.subplots(3, 2)
            fig.set_figheight(15)
            fig.set_figwidth(15)
            axes[0, 0].set_title("mean(intensity) from left end")
            im = axes[0, 0].imshow(heatmap_int_le, cmap="RdBu_r", norm=TwoSlopeNorm(0))
            axes[1, 0].set_title("mean(abs(sv)) from left end")
            im2 = axes[1, 0].imshow(heatmap_abs_le, cmap="RdBu_r", norm=TwoSlopeNorm(0))
            axes[2, 0].set_title("mean(sv) from left end")
            im3 = axes[2, 0].imshow(heatmap_le, cmap="RdBu_r", norm=TwoSlopeNorm(0))
            axes[0, 1].set_title("mean(intensity) from right end")
            im4 = axes[0, 1].imshow(heatmap_int_re, cmap="RdBu_r", norm=TwoSlopeNorm(0))
            axes[1, 1].set_title("mean(abs(sv)) from right end")
            im5 = axes[1, 1].imshow(heatmap_abs_re, cmap="RdBu_r", norm=TwoSlopeNorm(0))
            axes[2, 1].set_title("mean(sv) from right end")
            im6 = axes[2, 1].imshow(heatmap_re, cmap="RdBu_r", norm=TwoSlopeNorm(0))
            for ax in axes.flatten():
                ax.set_yticks(np.arange(len(self.amino_acids_sorted)))
                ax.set_yticklabels(self.amino_acids_sorted, size=6)
                ax.set_xticks(np.arange(30))
                ax.set_xticklabels(np.arange(30), size=6)
            fig.colorbar(im).ax.set_yscale("linear")
            fig.colorbar(im2).ax.set_yscale("linear")
            fig.colorbar(im3).ax.set_yscale("linear")
            fig.colorbar(im4).ax.set_yscale("linear")
            fig.colorbar(im5).ax.set_yscale("linear")
            fig.colorbar(im6).ax.set_yscale("linear")
            if save is not False:
                plt.savefig(save + "/position_heatmap.png", bbox_inches="tight")
            else:
                plt.show()

    def aa_heatmap(self, save=False):
        plt.close("all")
        fig, axes = plt.subplots(3, 4)
        fig.set_figheight(15)
        fig.set_figwidth(17)

        for ax in axes.flatten():
            ax.set_yticks(np.arange(len(self.amino_acids_sorted)))
            ax.set_yticklabels(self.amino_acids_sorted, size=6)
            ax.set_ylabel("AA(N)")
            ax.set_xticks(np.arange(len(self.amino_acids_sorted)))
            ax.set_xticklabels(self.amino_acids_sorted, size=6)
            ax.set_xlabel("AA(C)")

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
            for A, a in enumerate(self.amino_acids_sorted):
                for B, b in enumerate(self.amino_acids_sorted):
                    tok = "%c-%c" % (a, b)
                    if tok in self.combo_pos_sv_sum[i]:
                        if len(self.combo_pos_sv_sum[i][tok]) > MIN_OCCUR_HEAT:
                            heatmap[A, B] = np.mean(self.combo_pos_sv_sum[i][tok])
                            heatmap_abs[A, B] = np.mean(
                                self.combo_pos_sv_abs_sum[i][tok]
                            )
                            heatmap_int[A, B] = np.mean(self.combo_sv_sum[i][tok])

            axes[0, i].set_title("%d-%d: mean(intensity)" % tuple(combo))
            im = axes[0, i].imshow(heatmap_int, cmap="RdBu_r", norm=TwoSlopeNorm(0))
            axes[1, i].set_title("%d-%d: mean(sv)" % tuple(combo))
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
        # sorted_amino_acids_sv_keys = sorted(
        #     self.amino_acids_sv,
        #     key=lambda key: len(self.amino_acids_sv[key]),
        #     reverse=True,
        # )

        fig, axs = plt.subplots(ncols=2)
        fig.set_figwidth(15)
        fig.set_figheight(8)

        data_right = {"shap_value": [], "amino_acid": [], "position": []}
        for key, values in self.amino_acid_pos_from_right.items():
            data_right["shap_value"].extend(values)
            data_right["amino_acid"].extend([key[0]] * len(values))
            data_right["position"].extend([int(key[2:])] * len(values))

        plot_right = sns.stripplot(
            data=data_right,
            order=self.amino_acids_sorted,
            x="shap_value",
            y="amino_acid",
            size=3,
            jitter=0.4,
            hue="position",
            palette="RdBu_r",
            legend=False,
            ax=axs[0],
        )

        cmap_right = plt.get_cmap("RdBu_r")
        norm_right = plt.Normalize(0, max(data_right["position"]))
        sm_right = matplotlib.cm.ScalarMappable(norm=norm_right, cmap=cmap_right)
        sm_right.set_array([])
        cbar_right = fig.colorbar(sm_right, ax=plot_right, shrink=0.7)
        cbar_right.set_label("Position from right")

        data_left = {"shap_value": [], "amino_acid": [], "position": []}
        for key, values in self.amino_acid_pos_from_left.items():
            data_left["shap_value"].extend(values)
            data_left["amino_acid"].extend([key[0]] * len(values))
            data_left["position"].extend([int(key[2:])] * len(values))

        plot_left = sns.stripplot(
            data=data_left,
            order=self.amino_acids_sorted,
            x="shap_value",
            y="amino_acid",
            size=3,
            jitter=0.4,
            hue="position",
            palette="RdBu_r",
            legend=False,
            ax=axs[1],
        )

        cmap_left = plt.get_cmap("RdBu_r")
        norm_left = plt.Normalize(0, max(data_left["position"]))
        sm_left = matplotlib.cm.ScalarMappable(norm=norm_left, cmap=cmap_left)
        sm_left.set_array([])
        cbar_left = fig.colorbar(sm_left, ax=plot_left, shrink=0.7)
        cbar_left.set_label("Position from left")

        if save is not False:
            plt.savefig(save + "/swarmplot.png", bbox_inches="tight")
        else:
            plt.show()

    def full_report(self, save="."):
        self.aa_only_plot(save=save)
        self.position_only_plot(save=save)
        self.position_heatmap(save=save)
        self.aa_heatmap(save=save)
        self.swarmplot(save=save)


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as file:
        config = yaml.safe_load(file)["shap_visualization"]
    visualization = ShapVisualization(config["sv_path"])
    visualization.full_report(save=str(Path(config["sv_path"]).parent.absolute()))
    # visualization.clustering(config)
