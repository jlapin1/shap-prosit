import numpy as np
import pandas as pd
import dlomix
from dlomix import constants, data, eval, layers, models, pipelines, reports, utils
import tensorflow as tf
import sys

##########################################
#               Load Data                #
##########################################

from dlomix.data import IntensityDataset

TRAIN_DATAPATH = './intensity_data.csv'

BATCH_SIZE = 64
SEQ_LEN = 30

#np.random.seed(10)
#tf.random.set_seed(10)
intdata = IntensityDataset(
    data_source=TRAIN_DATAPATH, 
    seq_length=SEQ_LEN, 
    batch_size=BATCH_SIZE, 
    val_ratio=0.2, 
    test=False
)

# Enumerate and map all possible inputs
lst = []
for i in intdata.train_data:
    for j in range(i[0]['sequence'].shape[0]):
        lst.append([ord(m) for m in i[0]['sequence'].numpy()[j] if len(m)>0])
lst = [m for n in lst for m in n]
dic = {m: b'%c'%m for m in np.unique(lst)}
dic[1000] = b''
rev_dic = {n:m for m,n in dic.items()}

print("Training examples", BATCH_SIZE * len(intdata.train_data))
print("Validation examples", BATCH_SIZE * len(intdata.val_data))

###########################################
#             Custom loss                 #
###########################################
K = tf.keras
L = K.losses

def cs_loss(y_true, y_pred, sqrt=True, clip=True):
    if clip:
        y_true = tf.clip_by_value(y_true, 0, 1)
        y_pred = tf.clip_by_value(y_pred, 0, 1)
    if sqrt:
        cond = tf.where(y_true>0)
        gathered = tf.gather_nd(y_true, cond)
        y_true = tf.tensor_scatter_nd_update(y_true, cond, tf.sqrt(gathered))
        cond = tf.where(y_pred>0)
        gathered = tf.gather_nd(y_pred, cond)
        y_pred = tf.tensor_scatter_nd_update(y_pred, cond, tf.sqrt(gathered))
    loss = L.cosine_similarity(y_true, y_pred, axis=-1)
    loss = tf.reduce_mean(loss)
    return loss

###########################################
#                Model                    #
###########################################

from dlomix.models import PrositIntensityPredictor

class KerasActLayer(K.layers.Layer):
    def __init__(self, act):
        super(KerasActLayer, self).__init__()
        self.act = act
    def call(self, x):
        return self.act(x)

model = PrositIntensityPredictor(seq_length=SEQ_LEN)
latest = tf.train.latest_checkpoint("saved_model")
model.load_weights(latest)


# ###########################################
# #             Training                    #
# ###########################################
# from dlomix.losses import masked_spectral_distance

# model.compile(
#     optimizer='adam', 
#     loss=masked_spectral_distance, 
# )

# history = model.fit(intdata.train_data, validation_data=intdata.val_data, epochs=20)

# model.save_weights('saved_model/savedmodel')


###########################################
#             SHAP code                   #
###########################################

import shap

def annotations():
    ions = {}
    count=0
    for i in np.arange(1,30,1):
        for j in ['y','b']:
            for k in [1,2,3]:
                ion = '%c%d+%d'%(j,i,k)
                ions[ion] = count
                count+=1

    return ions

class ShapCalculator:
    def __init__(self, ion, dset, bgd, max_sequence_length=30, max_charge=6):
        self.val = dset
        self.bgd = bgd
        self.max_len = max_sequence_length
        self.max_charge = max_charge

        self.bgd_sz = bgd.shape[0]
        self.ion_ind = annotations()[ion]
        self.ext = int(ion[1])
        self.fnull = np.array([
            model(self.hx(bgd), training=False)[:, self.ion_ind].numpy().squeeze().mean()
        ])

        self.savepep = []
        self.savecv = []

    @tf.function
    def mask_pep(self, zs, pep, bgd_inds, mask=True):
        out = tf.zeros((tf.shape(zs)[0], tf.shape(pep)[1]), dtype=tf.string)
        if mask:
            ### TF
            ## Collect all peptide tokens that are 'on' and place them in the out tensor
            oneinds = tf.where(zs==1)
            onetokens = tf.gather_nd(tf.tile(pep, [tf.shape(zs)[0], 1]), oneinds)
            out = tf.tensor_scatter_nd_update(out, oneinds, onetokens)
            ## Replace all peptide tokens that are 'off' with background dataset
            zeroinds = tf.where(zs==0)
            # Random permutation of BGD peptides
            #randperm = tf.random.uniform_candidate_sampler(
            #    tf.ones((tf.shape(zs)[0], BGD_SZ), dtype=tf.int64),
            #    num_true=BGD_SZ, num_sampled=tf.shape(zs)[0], unique=True, range_max=BGD_SZ
            #)[0][:,None]
            #randperm = tf.random.uniform((tf.shape(zs)[0], 1), 0, BGD_SZ, dtype=tf.int32)
            bgd_ = tf.gather_nd(self.bgd, bgd_inds[:,None])
            # gather specific tokens of background dataset that belong to 'off' inds
            bgd_ = tf.gather_nd(bgd_, zeroinds)
            # Place the bgd tokens in the out tensor
            out = tf.tensor_scatter_nd_update(out, zeroinds, tf.reshape(bgd_, (-1,)))
            # pad c terminus with b''
            inds2 = tf.cast(tf.argmax(tf.equal(out, b''), 1), tf.int32)

            #nok = tf.where(inds2==0)
            #amt = tf.shape(nok)[0] #tf.reduce_sum(tf.cast(nok, tf.int32))
            #inds2 = tf.tensor_scatter_nd_update(
            #    inds2, nok, tf.shape(pep)[1]*tf.ones((amt,), dtype=tf.int32)
            #)
            inds2_ = tf.tile(tf.range(self.max_len, dtype=tf.int32)[None], [tf.shape(out)[0], 1])
            inds1000 = tf.where(inds2_ > inds2[:,None])
            out = tf.tensor_scatter_nd_update(
                out, inds1000, tf.fill((tf.shape(inds1000)[0],), b'')
            )
        else:
            out = pep

        #self.savepep.append(out)
        #self.savecv.append(zs)
        return out

    @tf.function
    def hx(self, tokens):

        sequence = tokens[:,:-2]
        collision_energy = tf.strings.to_number(tokens[:,-2:-1])
        precursor_charge = tf.one_hot(
            tf.cast(tf.strings.to_number(tokens[:,-1]), tf.int32)-1,
            self.max_charge
        )
        z = {
            'sequence': sequence,
            'collision_energy': collision_energy,
            'precursor_charge': precursor_charge
        }

        return z

    def EnsPred(self, pep, batsz=100, mask=True):
        # pep: coalition vectors, 1s and 0s; excludes absent AAs
        P = pep.shape

        # Chunk into batches, each <= batsz
        batches = (
            np.split(pep, np.arange(batsz, batsz*(P[0]//batsz), batsz), 0)
            if P[0] % batsz==0 else
            np.split(pep, np.arange(batsz, batsz*(P[0]//batsz)+1, batsz), 0)
        )

        # Use these indices to substitute values from background dataset
        # - bgd sample is run for each coalition vector
        rpts = P[0] // self.bgd_sz + 1 # number of repeats
        bgd_indices = tf.concat(rpts*[tf.range(self.bgd_sz, dtype=tf.int32)], 0)

        out_ = []
        for I, batch in enumerate(batches):

            # AAs (cut out CE, charge)
            # Absent AAs (all 1s)
            # [CE, CHARGE]
            batch = np.concatenate([
                batch[:,:-2],
                np.ones((tf.shape(batch)[0], self.max_len - tf.shape(pep)[1] + 2)),
                batch[:,-2:]
            ], axis=1)
            batch = tf.constant(batch, tf.int32)

            # Indices of background dataset to use for subbing in 0s
            bgd_inds = bgd_indices[I*batsz : (I+1)*batsz][:tf.shape(batch)[0]]

            # Create 1/0 mask and then turn into model ready input
            inp = self.hx(self.mask_pep(batch, self.inp_orig, bgd_inds, mask))

            # Run through model
            out = model(inp, training=False)
            out_.append(out)

        out_ = tf.concat(out_, axis=0)

        return out_

    def Score(self, peptide, mask=True):
        shape = tf.shape(peptide)
        x_ = self.EnsPred(peptide, mask=mask)[:, self.ion_ind]
        score = tf.squeeze(x_).numpy()
        if shape[0] == 1:
            score = np.array([score])[None, :]

        return score

    def calc_shap_values(self, index, samp=1000):

        # String array
        inp_orig = self.val[index : index+1]
        self.inp_orig = inp_orig

        # Peptide length for the current peptide
        pl = sum(inp_orig[0] != '') - 2
        if pl<=self.ext:
            return False

        # Input coalition vector: All aa's on (1) + charge + eV
        # - Padded amino acids are added in as all ones (always on) in EnsPred
        inpvec = np.ones((1, pl+2))

        # Mask vector is peptide length all off
        # - By turning charge and eV on, I am ignoring there contribution
        maskvec = np.zeros((self.bgd_sz, pl+2))
        maskvec[:,-2:] = 1

        orig_spec = self.EnsPred(inpvec, mask=False)[:, self.ion_ind]

        # SHAP Explainer
        ex = shap.KernelExplainer(self.Score, maskvec)
        ex.fnull = self.fnull
        ex.expected_value = ex.fnull

        # Calculate the SHAP values
        seq = list(inp_orig.squeeze())
        seqrep = ' '.join(seq[:pl]) + ' ' + ' '.join(seq[-2:])
        # print(seqrep)
        inten = float(orig_spec.numpy().squeeze())
        # print("Calculated intensity: %f"%inten)
        # print("fnull: %f"%ex.fnull)
        # print("Expectation value: %f"%ex.expected_value)
        samp_ = Samp if pl<20 else 5*Samp
        shap_values = ex.shap_values(inpvec, nsamples=Samp)

        # for i,j in zip(seq, shap_values.squeeze()[:pl]):
            # print('%c: %10f'%(i,j))
        # print(np.sum(shap_values))

        return {
            'int': inten,
            'sv': shap_values.squeeze()[:pl],
            'seq': seqrep
        }

    def write_shap_values(self, out_dict, path='output.txt'):
        inten = out_dict['int']
        shap_values = out_dict['sv']
        seqrep = out_dict['seq']
        with open(path, 'a') as f:
            f.write(seqrep + ' %f\n'%inten)
            f.write(' '.join(['%s'%m for m in shap_values.squeeze()])+'\n')

###################################################################################
#                               Setttings                                         #
###################################################################################

BGD_SZ = 100
Samp = 1000
VAL_INDEX = 1 # 28, 5509, 5812, 6384
ION = 'y5+1'
WRITE = True

# Background dataset
"""
THIS IS THE CODE THAT PRODUCED THE BACKGROUND DATASET IN val_inps.csv
inps = []
for i in intdata.val_data:
    charges = i[0]['precursor_charge'].numpy().argmax(1) + 1
    for j in range(i[0]['sequence'].shape[0]):
        csseq = ",".join([k.numpy().decode('utf-8') for k in i[0]['sequence'][j]])
        ce = i[0]['collision_energy'][j].numpy()[0]
        inp = csseq + ',%.2f,%d'%(ce, charges[j])
        inps.append(inp)
with open("val_inps.csv", 'w') as f:
    f.write("\n".join(inps))
"""
val = np.array([m.split(',') for m in open('val_inps.csv').read().split("\n")])
#perm = np.random.permutation(np.arange(len(val)))
#np.savetxt("perm.txt", perm, fmt='%d')
perm = np.loadtxt('perm.txt').astype(int)

bgd = val[perm[:BGD_SZ]]
val = val[perm[BGD_SZ:]]

sc = ShapCalculator(ION, val, bgd)

for INDEX in range(int(sys.argv[1]), int(sys.argv[1])+int(sys.argv[2]), 1):#val.shape[0], 1):
    print("\r%d/%d"%(INDEX, len(val)), end='\n')

    out_dict = sc.calc_shap_values(INDEX, samp=Samp)
    if WRITE & (out_dict!=False):
        sc.write_shap_values(out_dict)
