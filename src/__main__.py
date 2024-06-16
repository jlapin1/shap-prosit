import os
import sys

import yaml

from src import ShapVisualization, save_shap_values
from src.models.model_wrappers import model_wrappers

if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config_calculator = config["shap_calculator"]

    if not os.path.exists(config_calculator["ion"]):
        os.makedirs(config_calculator["ion"])

    model_wrapper = model_wrappers[config_calculator["model_type"]](
        path=config_calculator["model_path"], ion=config_calculator["ion"]
    )

    save_shap_values(
        val_data_path=config_calculator["val_inps_path"],
        model_wrapper=model_wrapper,
        ion=config_calculator["ion"],
        perm_path=config_calculator["perm_path"],
        output_path=config_calculator["ion"],
        samp=config_calculator["samp"],
        bgd_sz=config_calculator["bgd_sz"],
    )

    visualization = ShapVisualization(config_calculator["ion"] + "/output.parquet.gzip")
    visualization.full_report(save=config_calculator["ion"])
    visualization.clustering(config["shap_visualization"])
