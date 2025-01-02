import pkg_resources
installed_packages = pkg_resources.working_set
package_key = [m for m in installed_packages.entry_keys.keys() if 'site-packages'==m.split('/')[-1]][0]
installed_packages = installed_packages.entry_keys[package_key]
if 'torch' in installed_packages:
    from .peptide_encoder import PeptideEncoderModel
    from .prosit import PrositModel
