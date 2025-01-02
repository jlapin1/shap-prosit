import pkg_resources
installed_packages = pkg_resources.working_set
installed_packages = installed_packages.entry_keys[
    '/cmnfs/home/j.lapin/miniconda3/envs/shap/lib/python3.10/site-packages'
]
if 'torch' in installed_packages:
    from .peptide_encoder import PeptideEncoderModel
    from .prosit import PrositModel
