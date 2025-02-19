import os
import sys
from operator import itemgetter
from pathlib import Path
from typing import Union, List
import re

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from statistics import mean
from collections import defaultdict

import yaml
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.cluster import Birch, KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

matplotlib.use("Agg")
MIN_OCCUR_AVG = 100
MIN_OCCUR_HEAT = 15

ion_extent = lambda string: int(string[1:].split('+')[0].split('^')[0])

reverse_ion_list = ["x", "X", "y", "z", "Z"]

amino_acid_list = list("AVILMFYWSTNQCUGPRHKDEU")

def contains_amino_acid(token):
    # remove modiffication, if necessary
    token_ = re.sub(r"\[UNIMOD:[0-9]{1,3}]", '', token)
    return True if token_ in amino_acid_list else False

class ShapVisualizationIntensity:
    def __init__(
        self,
        sv_path: Union[str, bytes, os.PathLike],
        ion: str,
        position_combos: List[list] | None = None,
        filter_expr: str = None,
        bitoken: bool = False,
        maximum_sequence_length: int = 30,
    ) -> None:
        self.maximum_sequence_length = maximum_sequence_length
        self.default_position_combos = [
            [0, -1],
            [ion_extent(ion) - 1, 0],
            [ion_extent(ion) - 1, -1],
        ]
        self.position_combos = position_combos
        df = pd.read_parquet(sv_path)
        df["sequence_length"] = np.vectorize(len)(df["sequence"])

        # Grab only the intensity and shap column for the ion of interest
        int_col = f"intensity_{ion}"
        assert int_col in df.columns, f"Available columns {df.columns}"
        sv_col = f"shap_values_{ion}"
        assert sv_col in df.columns
        bgd_col = f"bgd_mean_{ion}"
        assert bgd_col in df.columns
        df = df[['sequence', 'energy', 'charge', 'method', 'sequence_length', bgd_col, int_col, sv_col]]
        df = df.rename(columns={bgd_col: 'bgd_mean', int_col: 'intensity', sv_col: 'shap_values'})
        # Automatic filter based on sequences for which ion couldn't be predicted
        df = df[df['intensity'] != -1]
        
        # Custom filter (from config)
        if filter_expr is not None:
            df = df.query(filter_expr)
        self.ion = ion
        
        # Get data from dataframe
        self.pred_intensities = df["intensity"].tolist()
        self.charge = df["charge"].tolist()
        self.energy = df["energy"].tolist()
        self.seq_list = df["sequence"].tolist()
        self.shap_values_list = df["shap_values"].tolist()
        self.bgd_mean = df["bgd_mean"].tolist()[0]

        dataframe = {
            'original': [], 
            'abbreviated': [],
            'token': [],
            'amino_acid': [],
            'position': [],
            'modification': [],
            'shap_values': [],
            'intensities': [],
        }
        list_index = {}

        dataframe2 = {
            'token1': [],
            'token2': [],
            'amino_acid1': [],
            'amino_acid2': [],
            'abbreviated1': [],
            'abbreviated2': [],
            'position1': [],
            'position2': [],
            'bitoken': [],
            #'shap_values1': [],
            #'shap_values2': [],
            'mean_shap_values': [],
            'abs_shap_values': [],
            'intensities': [],
        }
        list_index2 = {}
        
        total = len(self.seq_list)
        pbar = tqdm(zip(self.seq_list, self.shap_values_list, self.pred_intensities), total=total)
        for sequence, shap_values, intensity in pbar:
            
            seq = np.array(sequence)
            sv = np.array(shap_values)
            inten = intensity
            
            # aa_position has number amino acid at every position, which will not
            # be equal to absolute position if non-amino acid tokens exist
            hold = np.array(list(map(contains_amino_acid, seq)))
            aa_position = np.maximum(np.cumsum(hold) - 1, 0)
            aa_max_pos = aa_position.max() + 1
            #token_position = np.arange(len(seq))
            #le = len(seq)
            #self.count_positions += np.append(np.ones((le)), np.zeros((maximum_sequence_length - le)))
            
            for i, (am_ac, aa_pos, sh_value) in enumerate(zip(seq, aa_position, sv)):

                relative_position = (
                    aa_pos + ion_extent(self.ion) - aa_max_pos
                    if self.ion[0] in reverse_ion_list else
                    ion_extent(self.ion) - aa_pos - 1
                )
                
                abbr = re.sub("\[UNIMOD:|\]", "", am_ac)
                token = f"{abbr}_{relative_position}"

                if token not in dataframe['token']:
                    dataframe['original'].append(am_ac)
                    dataframe['abbreviated'].append(abbr)
                    dataframe['token'].append(token)
                    AA = re.sub(r"\[UNIMOD:[0-9]{1,3}|\]", "", am_ac)
                    dataframe['amino_acid'].append(AA)
                    dataframe['position'].append(relative_position)
                    mod_number = re.sub(r"[A-Z]|\[UNIMOD:|\]", "", am_ac)
                    mod_number = int(mod_number) if mod_number != '' else -1
                    dataframe['modification'].append(mod_number)
                    lind = len(dataframe['shap_values'])
                    list_index[token] = lind
                    dataframe['shap_values'].append([])
                    dataframe['intensities'].append([])
                dataframe['shap_values'][list_index[token]].append(sh_value)
                dataframe['intensities'][list_index[token]].append(inten)
                
                if bitoken:
                    for j, (am_ac2, aa_pos2, sh_value2) in enumerate(zip(seq, aa_position, sv)):
                        if i != j:
                            relative_position2 = (
                                aa_pos2 + ion_extent(self.ion) - aa_max_pos
                                if self.ion[0] in reverse_ion_list else
                                ion_extent(self.ion) - aa_pos2 - 1
                            )
                            
                            abbr2 = re.sub("\[UNIMOD:|\]", "", am_ac2)
                            pos = np.array([relative_position, relative_position2])
                            amac = np.array([am_ac, am_ac2])
                            abbrs = np.array([abbr, abbr2])
                            sort = np.argsort([relative_position, relative_position2])
                            pos = pos[sort]
                            # Check to see if any combos are fulfilled
                            proceed = any([
                                relative_position==combo[0] and relative_position2==combo[1] 
                                for combo in position_combos
                            ]) if position_combos is not None else True
                            if proceed:
                                amac = amac[sort]
                                abbrs = abbrs[sort]
                                token1 = f"{abbrs[0]}_{pos[0]}"
                                token2 = f"{abbrs[1]}_{pos[1]}"
                                Bitoken = f"{token1}|{token2}"
                                #if token2 == 'P_0':
                                #    print()
                                if Bitoken not in dataframe2['bitoken']:
                                    dataframe2['bitoken'].append(Bitoken)
                                    dataframe2['token1'].append(token1)
                                    dataframe2['token2'].append(token2)
                                    dataframe2['abbreviated1'].append(abbrs[0])
                                    dataframe2['abbreviated2'].append(abbrs[1])
                                    dataframe2['amino_acid1'].append(amac[0])
                                    dataframe2['amino_acid2'].append(amac[1])
                                    dataframe2['position1'].append(pos[0])
                                    dataframe2['position2'].append(pos[1])
                                    lind = len(dataframe2['mean_shap_values'])
                                    list_index2[Bitoken] = lind
                                    #dataframe2['shap_values1'].append([])
                                    #dataframe2['shap_values2'].append([])
                                    dataframe2['mean_shap_values'].append([])
                                    dataframe2['abs_shap_values'].append([])
                                    dataframe2['intensities'].append([])
                                #dataframe2['shap_values1'][list_index2[Bitoken]].append(sh_value)
                                #dataframe2['shap_values2'][list_index2[Bitoken]].append(sh_value2)
                                dataframe2['mean_shap_values'][list_index2[Bitoken]].append((sh_value+sh_value2)/2)
                                dataframe2['abs_shap_values'][list_index2[Bitoken]].append(abs(sh_value)+abs(sh_value2))
                                dataframe2['intensities'][list_index2[Bitoken]].append(inten)
        
        hold = pd.DataFrame(dataframe).set_index('token')
        hold['occurs'] = hold.apply(lambda x: len(x['shap_values']), axis=1)
        hold['sv_mean'] = hold.apply(lambda x: np.mean(x['shap_values']), axis=1)
        hold['sv_abs_mean'] = hold.apply(lambda x: np.mean(np.abs(x['shap_values'])), axis=1)
        hold['sv_std'] = hold.apply(lambda x: np.std(x['shap_values']), axis=1)
        hold['int_mean'] = hold.apply(lambda x: np.mean(x['intensities']), axis=1)
        self.tokenframe = hold
        
        if bitoken:
            hold2 = pd.DataFrame(dataframe2).set_index('bitoken')
            hold2['occurs'] = hold2.apply(lambda x: len(x['mean_shap_values']), axis=1)
            hold2['sv_mean_mean'] = hold2.apply(lambda x: np.mean(x['mean_shap_values']), axis=1)
            hold2['sv_abs_mean'] = hold2.apply(lambda x: np.mean(x['abs_shap_values']), axis=1)
            hold2['sv_mean_std'] = hold2.apply(lambda x: np.std(x['mean_shap_values']), axis=1)
            hold2['sv_abs_std'] = hold2.apply(lambda x: np.std(x['abs_shap_values']), axis=1)
            hold2['int_mean'] = hold2.apply(lambda x: np.mean(x['intensities']), axis=1)
            self.bitokenframe = hold2
        
    """
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
            visualization = ShapVisualizationIntensity(
                str(Path(config["sv_path"]).parent.absolute())
                + f"/cluster_{cluster}/output.parquet.gzip",
                self.ion,
                [
                    [0, number_of_aas - 1],
                    [1, number_of_aas - 2],
                    [0, 1],
                    [number_of_aas - 2, number_of_aas - 1],
                ],
                filter_expr=config["filter_expr"],
            )
            visualization.full_report(
                save=str(Path(config["sv_path"]).parent.absolute())
                + f"/cluster_{cluster}"
            )
    """

    def aa_only_plot(self, save=False):
        plt.close("all")
        fig, axes = plt.subplots(4, 1)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        axes[0].set_title("Mean absolute shapley values for ion " + self.ion)
        axes[0].set_xlim([-0.5, 19.5])
        sorted_aas = sorted(self.tokenframe.query("amino_acid != ''")['abbreviated'].unique())
        meanabs = [
            np.mean([abs(m) for n in self.tokenframe.query(f"abbreviated == '{aa}'")['shap_values'].tolist() for m in n])
            for aa in sorted_aas
        ]
        axes[0].plot(meanabs, "ro")
        im = axes[1].imshow(
            np.array(meanabs)[None],
            cmap="RdBu_r",
            norm=TwoSlopeNorm(0),
        )
        axes[2].set_title("Mean shapley values for ion " + self.ion)
        mean = [
            np.mean([m for n in self.tokenframe.query(f"abbreviated == '{aa}'")['shap_values'].tolist() for m in n])
            for aa in sorted_aas
        ]
        std = [
            np.std([m for n in self.tokenframe.query(f"abbreviated == '{aa}'")['shap_values'].tolist() for m in n])
            for aa in sorted_aas
        ]
        axes[2].errorbar(
            np.arange(len(sorted_aas)),
            np.array(mean),
            np.array(std),
            marker="o",
            linestyle="none",
            markerfacecolor="red",
            markersize=10,
        )
        im2 = axes[3].imshow(
            np.array(mean)[None],
            cmap="RdBu_r",
            norm=TwoSlopeNorm(0),
        )
        axes[3].set_xlim([-0.5, 19.5])
        for ax in axes:
            ax.set_xticks(np.arange(len(sorted_aas)))
            ax.set_xticklabels(sorted_aas, size=8, rotation=270)
        for ax in axes[[1, 3]]:
            ax.set_yticks([])
            ax.set_yticklabels([])
        fig.colorbar(im, ax=axes[:2]).ax.set_yscale("linear")
        fig.colorbar(im2, ax=axes[2:]).ax.set_yscale("linear")
        if save is not False:
            plt.savefig(os.path.join(save, "aa_only_plot.png"), bbox_inches="tight")
        else:
            plt.show()

    def position_only_plot(self, save=False):
        plt.close("all")
        fig, axes = plt.subplots(2)
        fig.set_figheight(5)
        fig.set_figwidth(15)
        axes[0].set_title("Mean absolute shapley values for ion " + self.ion)
        Range = np.arange(self.tokenframe['position'].min(), self.tokenframe['position'].max()+1, 1)
        meanabs = np.array([
            np.mean([abs(m) for n in self.tokenframe.query(f"position == {pos}")['shap_values'].tolist() for m in n])
            for pos in Range
        ])
        im = axes[0].imshow(meanabs[None], cmap="RdBu_r", norm=TwoSlopeNorm(0))
        axes[1].set_title("Mean shapley values for ion " + self.ion)
        mean = np.array([
            np.mean([m for n in self.tokenframe.query(f"position == {pos}")['shap_values'].tolist() for m in n])
            for pos in Range
        ])
        im2 = axes[1].imshow(mean[None], cmap="RdBu_r", norm=TwoSlopeNorm(0))
        tick_range = np.arange(
            ion_extent(self.ion) - 1,
            -1 * (self.maximum_sequence_length - ion_extent(self.ion) + 1),
            -1,
        )
        for ax in axes:
            ax.axvline(
                x=ion_extent(self.ion) - 0.5, color="black", linewidth=3
            )
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks(np.arange(self.maximum_sequence_length))
            ax.set_xticklabels(tick_range, size=6)
        fig.colorbar(im).ax.set_yscale("linear")
        fig.colorbar(im2).ax.set_yscale("linear")
        if save is not False:
            plt.savefig(os.path.join(save,"position_only_plot.png"), bbox_inches="tight")
        else:
            plt.show()

    def position_heatmap(self, all_tokens=False, save=False):
        plt.close("all")
        
        amino_acids = sorted(
            self.tokenframe['abbreviated'].unique()
            if all_tokens else
            self.tokenframe.query("amino_acid != ''")['abbreviated'].unique()
        )

        heatmap_int = np.zeros((len(amino_acids), self.maximum_sequence_length))
        heatmap = np.zeros((len(amino_acids), self.maximum_sequence_length))
        heatmap_abs = np.zeros((len(amino_acids), self.maximum_sequence_length))
        
        for A, a in enumerate(amino_acids):
            for b in np.arange(self.maximum_sequence_length):
                tok = "%s_%d" % (a, -1 * (b + 1 - ion_extent(self.ion)))
                if tok in self.tokenframe.index:
                    if self.tokenframe.loc[tok]['occurs'] > MIN_OCCUR_HEAT:
                        heatmap_int[A, b] = self.tokenframe.loc[tok]['int_mean']
                        heatmap[A, b] = self.tokenframe.loc[tok]['sv_mean']
                        heatmap_abs[A, b] = self.tokenframe.loc[tok]['sv_abs_mean']#amino_acid_pos_abs_avg[tok]

        fig, axes = plt.subplots(3)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        axes[0].set_title("Mean intensity for ion " + self.ion)
        im = axes[0].imshow(heatmap_int, cmap="RdBu_r", norm=TwoSlopeNorm(0))
        axes[1].set_title("Mean absolute shapley values for ion " + self.ion)
        im2 = axes[1].imshow(heatmap_abs, cmap="RdBu_r", norm=TwoSlopeNorm(0))
        axes[2].set_title("Mean shapley values for ion " + self.ion)
        im3 = axes[2].imshow(heatmap, cmap="RdBu_r", norm=TwoSlopeNorm(0))

        tick_range = np.arange(
            ion_extent(self.ion) - 1,
            -1 * (self.maximum_sequence_length - ion_extent(self.ion) + 1),
            -1,
        )
        for ax in axes:
            ax.axvline(
                x=ion_extent(self.ion) - 0.5, color="black", linewidth=3
            )
            ax.set_yticks(np.arange(len(amino_acids)))
            ax.set_yticklabels(amino_acids, size=8)
            ax.set_xticks(np.arange(self.maximum_sequence_length))
            ax.set_xticklabels(tick_range, size=7)
        fig.colorbar(im, pad=0.01).ax.set_yscale("linear")
        fig.colorbar(im2, pad=0.01).ax.set_yscale("linear")
        fig.colorbar(im3, pad=0.01).ax.set_yscale("linear")
        if save is not False:
            plt.savefig(os.path.join(save, "position_heatmap.png"), bbox_inches="tight")
        else:
            plt.show()
    
    def aa_heatmap(self, all_tokens=False, save=False):
        plt.close("all")
        
        position_combos = (
            self.default_position_combos if 
            self.position_combos == None
            else self.position_combos
        )
        
        if all_tokens:
            amino_acids = sorted(np.unique(self.bitokenframe['abbreviated1'].to_list() + self.bitokenframe['abbreviated2'].to_list()))
        else:
            all_aa = self.bitokenframe[(self.bitokenframe['abbreviated1']!='') | (self.bitokenframe['abbreviated2']!='')]
            amino_acids = sorted(np.unique(all_aa['abbreviated1'].to_list() + all_aa['abbreviated2'].to_list()))
        
        fig, axes = plt.subplots(3, len(self.position_combos))
        fig.set_figheight(15)
        fig.set_figwidth(17)

        for ax in axes.flatten():
            ax.set_yticks(np.arange(len(amino_acids)))
            ax.set_yticklabels(amino_acids, size=6)
            ax.set_ylabel("AA(L)")
            ax.set_xticks(np.arange(len(amino_acids)))
            ax.set_xticklabels(amino_acids, size=6, rotation=270)
            ax.set_xlabel("AA(R)")

        for i, combo in enumerate(self.position_combos):
            heatmap = np.zeros(
                (len(amino_acids), len(amino_acids))
            )
            heatmap_abs = np.zeros(
                (len(amino_acids), len(amino_acids))
            )
            heatmap_int = np.zeros(
                (len(amino_acids), len(amino_acids))
            )

            subset = self.bitokenframe.query(f"position1 == {combo[0]} and position2 == {combo[1]}")
            for l, aa1 in enumerate(amino_acids):
                for m, aa2 in enumerate(amino_acids):
                    subset_ = subset.query(f"abbreviated1 == '{aa1}' and abbreviated2 == '{aa2}'")
                    assert (len(subset_) > 1) == False
                    if len(subset_) == 1:
                        entry = subset_.iloc[0]
                        if entry['occurs'] > MIN_OCCUR_HEAT:
                            heatmap[l, m] = entry['sv_mean_mean']
                            heatmap_abs[l, m] = entry['sv_abs_mean']
                            heatmap_int[l, m] = entry['int_mean']

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
            plt.savefig(os.path.join(save, "aa_heatmap.png"), bbox_inches="tight")
        else:
            plt.show()

    def swarmplot(self, save=False):
        plt.close("all")
        fig, ax = plt.subplots()
        fig.set_figwidth(15)
        fig.set_figheight(13)
        
        data = {"shap_value": [], "amino_acid": [], "inside_ion": []}
        sorted_aas = sorted(self.tokenframe.query("amino_acid != ''")['abbreviated'].unique())
        for aaa in sorted_aas:
            total_occurs = self.tokenframe.query(f'abbreviated == "{aaa}"')['occurs'].sum()
            shap_values = [
                m for n in self.tokenframe.query(f'abbreviated == "{aaa}"')['shap_values'] 
                for m in n
            ]
            inside = [
                m for n in [
                    o['occurs']*[o['position']>=0] for _,o in 
                    self.tokenframe.query(f'abbreviated == "{aaa}"')[['position','occurs']].iterrows()
                ] 
                for m in n
            ]
            amino_acids = total_occurs * [aaa]
            
            # Permute so that order doesn't matter as colors cover each other up
            perm = np.random.permutation(np.array([shap_values, amino_acids, inside]).T)
            [shap_values, amino_acids, inside] = np.split(perm, 3, 1)
            
            SV = shap_values.astype(np.float32).squeeze()
            AA = amino_acids.squeeze()
            INSIDE = inside.squeeze()
            if len(SV.shape)<1:
               SV = SV[None]
               AA = AA[None]
               INSIDE = INSIDE[None]
            data['shap_value'].extend(SV.tolist())
            data['amino_acid'].extend(AA.tolist())
            data['inside_ion'].extend(INSIDE.tolist())

        plot = sns.stripplot(
            data=data,
            order=sorted_aas,
            x="shap_value",
            y="amino_acid",
            size=2,
            jitter=0.4,
            hue="inside_ion",
            legend=True,
            ax=ax,
        )
        plt.axvline(x=0, color="black", linewidth=4)

        if save is not False:
            plt.savefig(os.path.join(save, "swarmplot.png"))
        else:
            plt.show()

    def boxplot_position(self, exclude=['R','H','K'], save="."):
        plt.close("all")
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_figwidth(15)
        fig.set_figheight(10)
        min_occur_query = f'occurs >= {MIN_OCCUR_HEAT}'

        # All plot

        data = {"SHAP values": [], "Amino acid - Position": []}
    
        top20 = self.tokenframe.query(min_occur_query).iloc[
            self.tokenframe.query(min_occur_query)['sv_abs_mean'].argsort()[::-1]
        ].iloc[:20]
        for aaa in top20.index:
            data["SHAP values"].extend(top20.loc[aaa]['shap_values'])
            data["Amino acid - Position"].extend(top20.loc[aaa]['occurs'].sum()*[aaa])
        df = pd.DataFrame(data)
        sns.boxplot(
            ax=ax1,
            data=df,
            x="Amino acid - Position",
            y="SHAP values",
            color="#1f77b4",
        ).set_title("Shapley values per amino acid-position for ion " + self.ion)

        # Excldue plot

        data = {"SHAP values": [], "Amino acid - Position": []}
        
        wo_exclude = self.tokenframe[self.tokenframe['abbreviated'].str.startswith(tuple(exclude))==False]
        top20 = wo_exclude.query(min_occur_query).iloc[
            wo_exclude.query(min_occur_query)['sv_abs_mean'].argsort()[::-1]
        ].iloc[:20]
        for aaa in top20.index:
            data["SHAP values"].extend(top20.loc[aaa]['shap_values'])
            data["Amino acid - Position"].extend(top20.loc[aaa]['occurs'].sum()*[aaa])
        df = pd.DataFrame(data)
        sns.boxplot(
            data=df,
            x="Amino acid - Position",
            y="SHAP values",
            color="#1f77b4",
        ).set_title(
            "Shapley values per amino acid-position for ion "
            + self.ion
            + f" excluding {exclude} amino acids"
        )

        if save is not False:
            plt.savefig(os.path.join(save, "boxplot_position.png"), bbox_inches="tight")
        else:
            plt.show()
    
    """
    def boxplot_token(self, save="."):
        plt.close("all")
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_figwidth(15)
        fig.set_figheight(10)

        sum_abs_sv = {}

        for key in self.combo_pos_sv_sum[0].keys():
            if len(self.combo_pos_sv_sum[0][key]) < MIN_OCCUR_HEAT:
                continue
            sum_abs_sv[key] = mean([abs(x) for x in self.combo_pos_sv_sum[0][key]])

        data = {"SHAP values": [], "Amino acids on positions 0:-1": []}

        if sum_abs_sv:
            for aa in list(
                dict(
                    sorted(sum_abs_sv.items(), key=lambda x: x[1], reverse=True)[:20]
                ).keys()
            ):
                aa_abb = re.sub('UNIMOD:', '', aa)
                for shap in self.combo_pos_sv_sum[0][aa]:
                    data["SHAP values"].append(shap)
                    data["Amino acids on positions 0:-1"].append(aa_abb)

            df = pd.DataFrame(data)
            sns.boxplot(
                ax=ax1,
                data=df,
                x="Amino acids on positions 0:-1",
                y="SHAP values",
                color="#1f77b4",
            ).set_title("mean(sv)")

        sum_abs_sv_without_p = {}

        for key in self.combo_pos_sv_sum[0].keys():
            if len(self.combo_pos_sv_sum[0][key]) < MIN_OCCUR_HEAT or "P" in key:
                continue
            sum_abs_sv_without_p[key] = mean(
                [abs(x) for x in self.combo_pos_sv_sum[0][key]]
            )

        data = {"SHAP values": [], "Amino acids on positions 0:-1": []}

        if sum_abs_sv_without_p:
            for aa in list(
                dict(
                    sorted(
                        sum_abs_sv_without_p.items(), key=lambda x: x[1], reverse=True
                    )[:20]
                ).keys()
            ):
                aa_abb = re.sub('UNIMOD:', '', aa)
                for shap in self.combo_pos_sv_sum[0][aa]:
                    data["SHAP values"].append(shap)
                    data["Amino acids on positions 0:-1"].append(aa_abb)

            df = pd.DataFrame(data)
            sns.boxplot(
                ax=ax2,
                data=df,
                x="Amino acids on positions 0:-1",
                y="SHAP values",
                color="#1f77b4",
            ).set_title("mean(sv) without Proline")

        if save is not False:
            plt.savefig(os.path.join(save, "boxplot_bi_token.png"), bbox_inches="tight")
        else:
            plt.show()
    """

    def full_report(self, save="."):
        if not os.path.exists(save):
            os.makedirs(save)
        self.aa_only_plot(save=save)
        self.position_only_plot(save=save)
        self.position_heatmap(save=save)
        self.aa_heatmap(save=save)
        self.swarmplot(save=save)
        self.boxplot_position(save=save)
        #self.boxplot_token(save=save)

class ShapVisualizationGeneral():
    def __init__(
        self,
        sv_path: Union[str, bytes, os.PathLike],
        mode: str,
    ) -> None:

        self.mode = mode
        df = pd.read_parquet(sv_path)
        self.charge = df["charge"].tolist()
        self.seq_list = df["sequence"].tolist()
        self.shap_values_list = df[f"shap_values_{self.mode}"].tolist()

        # Initialize data structures
        self.count_positions = np.zeros((30))
        self.sv_sum = np.zeros((30))
        self.sv_abs_sum = np.zeros((30))
        self.combo_pos_sv_sum = []
        self.combo_pos_sv_abs_sum = []
        self.amino_acids_sv = {}
        self.amino_acid_pos = {}
        self.amino_acid_pos_generic = {}

        for sequence, shap_values in zip(self.seq_list, self.shap_values_list):
            # single peptide sequence
            seq = np.array(sequence)
            # shap values per amino acid in that peptide sequence
            sv = np.array(shap_values)

            # length of single peptide sequence
            le = len(seq)
            # increment position count for is aa~
            self.count_positions += np.append(np.ones((le)), np.zeros((30 - le)))

            # Gather sum and abs sum of SV in each position
            self.sv_sum[:le] += sv
            self.sv_abs_sum[:le] += abs(sv)

            for i, (am_ac, sh_value) in enumerate(zip(seq, sv)):
                # Store SHAP value for each AA
                if am_ac not in self.amino_acids_sv:
                    self.amino_acids_sv[am_ac] = []
                self.amino_acids_sv[am_ac].append(sh_value)

                # Create token (~id for aa-position plots) for AA and position
                # 0 is the first index in ion (from mode... maybe change this later), positives are inside ion
                # negatives are outside of ion
                tok_c = (
                    f"{am_ac}_{i}"
                )

                # Check if token already in dictionary
                if tok_c not in self.amino_acid_pos:
                    self.amino_acid_pos[tok_c] = []
                if tok_c not in self.amino_acid_pos_generic:
                    self.amino_acid_pos_generic[tok_c] = []

                # Store values for token in list
                self.amino_acid_pos[tok_c].append(sh_value)
                self.amino_acid_pos_generic[tok_c].append(sum(sv))

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
        self.amino_acid_pos_generic = {
            tok: np.mean(self.amino_acid_pos_generic[tok])
            for tok in self.amino_acid_pos_generic.keys()
            if len(self.amino_acid_pos_generic[tok]) > MIN_OCCUR_HEAT
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

    def aa_only_plot(self, save=False):
        plt.close("all")
        fig, axes = plt.subplots(4, 1)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        axes[0].set_title("Mean of absolute SHAP values per amino acid")
        axes[0].set_xlim([-0.5, 19.5])
        axes[0].plot(list(self.amino_acids_abs_avg_sv.values()), "ro")
        im = axes[1].imshow(
            np.array(list(self.amino_acids_abs_avg_sv.values()))[None],
            cmap="RdBu_r",
            norm=TwoSlopeNorm(0),
        )
        axes[2].set_title("Mean of SHAP values per amino acid")
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
        axes[0].set_title("Mean of absolute SHAP values per position")
        im = axes[0].imshow(self.sv_abs_avg[None], cmap="RdBu_r", norm=TwoSlopeNorm(0))
        axes[1].set_title("Mean of SHAP values per position")
        im2 = axes[1].imshow(self.sv_avg[None], cmap="RdBu_r", norm=TwoSlopeNorm(0))
        for ax in axes:
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks(np.arange(30))
            ax.set_xticklabels(np.arange(30), size=6)
        fig.colorbar(im).ax.set_yscale("linear")
        fig.colorbar(im2).ax.set_yscale("linear")
        if save is not False:
            plt.savefig(save + "/position_only_plot.png", bbox_inches="tight")
        else:
            plt.show()

    def position_length_plot(self, save=False, absolute=False):
        plt.close("all")

        position_length_dict = defaultdict(list)
        for peptide_sequence in self.shap_values_list:
            position_length_dict[len(peptide_sequence)].append(peptide_sequence)

        for key in position_length_dict.keys():
            if absolute:
                position_length_dict[key] = np.mean(np.abs(np.array(position_length_dict[key])), axis=0)
            else:
                position_length_dict[key] = np.mean(np.array(position_length_dict[key]), axis=0)
            position_length_dict[key] = np.pad(position_length_dict[key], (0, 30 - key))
            #np.concatenate(position_length_dict[key], np.array([None] * (30 - key)))

        df = pd.DataFrame(position_length_dict)
        df = df.reindex(sorted(df.columns), axis=1)
        df.index += 1
        #min_abundancy = df[df > 0].count() >= 10
        #df = df.loc[:, min_abundancy]
        df = df.transpose()

        plt.figure(figsize=(24, 16))
        mask = df.isnull()
        if absolute:
            cmap = sns.color_palette('Blues', as_cmap=True)
        else:
            cmap = sns.color_palette('RdBu_r', as_cmap=True)
        cmap.set_bad(color="lightgray")
        # mask for zeros maybe? Add abundancy filter.
        ax = sns.heatmap(df, annot=True, cmap=cmap, mask=mask, linewidths=0.5, linecolor='gray', edgecolor='gray')
        #ax.invert_yaxis()

        if save is not False:
            plt.savefig(f"{save}/position_length_plot_abs={absolute}.png", bbox_inches="tight")
        else:
            plt.show()




    def position_heatmap(self, save=False):
        plt.close("all")
        heatmap_generic = np.zeros((len(self.amino_acids_sorted), 30))
        heatmap = np.zeros((len(self.amino_acids_sorted), 30))
        heatmap_abs = np.zeros((len(self.amino_acids_sorted), 30))

        for A, a in enumerate(self.amino_acids_sorted):
            for b in np.arange(30):
                tok = "%c_%d" % (a, b)
                if tok in self.amino_acid_pos_generic:
                    heatmap_generic[A, b] = self.amino_acid_pos_generic[tok]
                if tok in self.amino_acid_pos_abs_avg:
                    heatmap[A, b] = self.amino_acid_pos_avg[tok]
                    heatmap_abs[A, b] = self.amino_acid_pos_abs_avg[tok]

        fig, axes = plt.subplots(3)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        axes[0].set_title(
            f"Mean of {'retention time' if mode == 'rt' else 'collisional cross section'} per amino acid & position"
        )
        im = axes[0].imshow(heatmap_generic, cmap="RdBu_r", norm=TwoSlopeNorm(0))
        axes[1].set_title("Mean of absolute SHAP values per amino acid & position")
        im2 = axes[1].imshow(heatmap_abs, cmap="RdBu_r", norm=TwoSlopeNorm(0))
        axes[2].set_title("Mean of SHAP values per amino acid & position")
        im3 = axes[2].imshow(heatmap, cmap="RdBu_r", norm=TwoSlopeNorm(0))
        for ax in axes:
            ax.set_yticks(np.arange(len(self.amino_acids_sorted)))
            ax.set_yticklabels(self.amino_acids_sorted, size=6)
            ax.set_xticks(np.arange(30))
            ax.set_xticklabels(np.arange(30), size=6)
        fig.colorbar(im).ax.set_yscale("linear")
        fig.colorbar(im2).ax.set_yscale("linear")
        fig.colorbar(im3).ax.set_yscale("linear")
        if save is not False:
            plt.savefig(save + "/position_heatmap.png", bbox_inches="tight")
        else:
            plt.show()

    def relative_position_heatmap(self, save=False):
        plt.close("all")

        rel_pos_per_aa_shap = defaultdict(list)
        for peptide, shap_values in zip(self.seq_list, self.shap_values_list):

            peptide_len = len(peptide)
            for position, amino_acid in enumerate(peptide):
                rel_pos = position/peptide_len
                if rel_pos < 0 or rel_pos > 1:
                    print(rel_pos)
                rel_pos_per_aa_shap[amino_acid].append((rel_pos, shap_values[position]))

        rows = []
        for amino_acid, values in rel_pos_per_aa_shap.items():
            for pos, shap in values:
                rows.append([amino_acid, pos, shap])

        bins = np.linspace(0, 1, 21)

        df = pd.DataFrame(rows, columns=["amino acid", "relative position", "shap value"])
        df['binned position'] = (
            pd.cut(df['relative position'], bins=bins, labels=np.round(bins[:-1], 2), include_lowest=True)
        )

        heatmap_data = df.pivot_table(index='amino acid', columns='binned position', values='shap value',
                                      aggfunc='mean')
        plt.figure(figsize=(24, 16))
        sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, cbar_kws={'label': 'SHAP value'}, center=0)
        plt.title('Heatmap of SHAP Values by Amino Acid and Binned Relative Position')

        if save is not False:
            plt.savefig(save + "/relative_position_heatmap.png", bbox_inches="tight")
        else:
            plt.show()


    def boxplot_position(self, save="."):
        plt.close("all")
        fig = plt.gcf()
        fig.set_figwidth(15)

        sum_abs_sv = {}

        for key in self.amino_acid_pos.keys():
            #if key.startswith(("R", "H", "K")):
            #    continue
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
        ).set_title("Absolute SHAP values (with high abundance) per Amino acid and its Position")

        if save is not False:
            plt.savefig(save + "/boxplot_position.png", bbox_inches="tight")
        else:
            plt.show()

    def full_report(self, save="."):
        self.aa_only_plot(save=save)
        self.position_only_plot(save=save)
        self.position_heatmap(save=save)
        self.boxplot_position(save=save)
        self.position_length_plot(save=save)
        self.position_length_plot(save=save, absolute=True)
        self.relative_position_heatmap(save=save)

if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as file:
        config = yaml.safe_load(file)

    mode = config["shap_visualization"]["mode"]
    if mode in {"rt", "cc", "fly1", "fly2", "fly3", "fly4", "charge1", "charge2", "charge3", "charge4", "charge5", "charge6"}:
        visualization = ShapVisualizationGeneral(
            config["shap_visualization"]["sv_path"], mode=mode
        )
    else:
        visualization = ShapVisualizationIntensity(
            config["shap_visualization"]["sv_path"], 
            ion=mode,
            filter_expr=config["shap_visualization"]["filter_expr"],
            bitoken=config['shap_visualization']['bitoken'],
            position_combos=config['shap_visualization']['position_combos'],
        )
    
    """Full report"""
    # If no specified out_path, save in dataframe's directory
    if config['shap_visualization']['out_path'] is None:
        save_path = str(Path(config["shap_visualization"]["sv_path"]).parent.absolute())
    else:
        save_path = config['shap_visualization']['out_path']
        if os.path.exists(save_path):
            user = input("<<<WARNING>>> The out_path you specified already exists. Continue? (y/n) ")
            if user.lower() == 'n':
                sys.exit()
            else:
                user = input("Would you like to change the directory path to save? (y/n) ")
                if user.lower() == 'y':
                    save_path = input("Type the new directory path starting from shap-prosit directory\n>>> ")
    visualization.full_report(save=save_path)
    
    if config['shap_visualization']['clustering']['run']:
        visualization.clustering(config["shap_visualization"])

