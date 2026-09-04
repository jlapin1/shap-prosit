import sys
import os
import yaml
import torch
import importlib.util
from glob import glob
# Set this
PROJECT_ROOT = os.path.expanduser("~/projects/denovo_base")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _force_load_denovo_modules():
	"""Ensure denovo_base root and its submodules are loaded into sys.modules."""
	if PROJECT_ROOT in sys.path:
		sys.path.remove(PROJECT_ROOT)
	sys.path.insert(0, PROJECT_ROOT)

	# Force load models.seq2seq directly if normal import fails
	if "models.seq2seq" not in sys.modules:
		seq2seq_path = os.path.join(PROJECT_ROOT, "models", "seq2seq.py")
		if os.path.exists(seq2seq_path):
			spec = importlib.util.spec_from_file_location("models.seq2seq", seq2seq_path)
			seq2seq_module = importlib.util.module_from_spec(spec)
			sys.modules["models.seq2seq"] = seq2seq_module
			spec.loader.exec_module(seq2seq_module)

_force_load_denovo_modules()

from loader import LoaderObj
from models.seq2seq import Seq2SeqMDLM

def load_model(project_directory, regex_extension='weights/*high*wts', device=None):
	
    _force_load_denovo_modules()
    
    with open(os.path.join(project_directory, "yaml/config.yaml")) as stream:
        config = yaml.safe_load(stream)

    # token dictionary
    L = LoaderObj()
    amod_dic = L.create_sequence_dictionary(config['loader']['dictionary_path'])
    synonyms = config['loader']['synonyms']
    if synonyms is not None:
        for pair in synonyms:
            letter_a, letter_b = pair
            amod_dic = L.synonym(letter_a, letter_b, amod_dic)
    amod_dic_rev = L.reverse_dictionary(amod_dic)

    # Seq2Seq model
    mdlm_config = config['decoder_mdlm']
    diff_config = config['decoder_mdlm']['diffusion_config']
    config['decoder_diff']['diffusion_config']['pad_tok_id'] = amod_dic['X']
    config['decoder_diff']['diffusion_config']['resume_checkpoint'] = False
    config['decoder_diff']['diffusion_config']['sequence_len'] = config['pep_length'][1] + 1 # b/c of eos token
    config['decoder_diff']['model_config']['self_condition'] = diff_config['model']['self_condition']

    model = Seq2SeqMDLM(
        encoder_config     = config['encoder_dict'],
        decoder_config     = config['decoder_diff']['model_config'],
        diff_config        = diff_config,
        top_peaks          = config['top_peaks'],
        max_peptide_length = config['pep_length'][1],
        token_dict         = amod_dic,
        ensemble_config    = config['decoder_diff']['ensemble'],
        masses_path        = config['loader']['masses_path'],
    )
    model.reverse = config['loader']['reverse']

    wts_path = glob(os.path.join(project_directory, regex_extension))[0]
    model.load_state_dict(torch.load(wts_path, map_location=device, weights_only=False))
    if device:
        model.to(device)

    return model


if __name__ == '__main__':
    
    # Set this
    regex_extension = "weights/*high*wts"
    
    model = load_model("/cmnfs/proj/diffusion/experiments/2026-08-31_09-22-44", regex_extension=regex_extension, device=device)
    
    dummy = {
        'mz': torch.empty(10, 150, device=device).uniform_(100,2000).sort(dim=-1)[0],
        'ab': torch.empty(10, 150, device=device).uniform_(0,1),
        'charge': torch.empty(10, device=device).uniform_(2, 4).round().int(),
        'mass': torch.empty(10, device=device).uniform_(300, 1000),
        'length': torch.full((10,), 150, device=device).int()
    }
    out_dict = model.forward_eval(dummy, progress=True)
    print(out_dict)
