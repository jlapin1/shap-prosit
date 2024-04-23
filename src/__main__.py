import os
import sys

import yaml

from src import ShapVisualization, save_shap_values

if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as file:
        config = yaml.safe_load(file)["shap_calculator"]

    if not os.path.exists(config["ion"]):
        os.makedirs(config["ion"])

    save_shap_values(
        val_data_path=config["val_inps_path"],
        model_path=config["model_path"],
        ion=config["ion"],
        perm_path=config["perm_path"],
        output_path=config["ion"],
        samp=config["samp"],
        bgd_sz=config["bgd_sz"],
    )

    visualization = ShapVisualization(config["ion"] + "/output.txt")
    visualization.full_report(save=config["ion"])
    visualization.clustering(config["shap_visualization"])
