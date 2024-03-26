import pandas as pd
import numpy as np

"""
mean(abs(sv)) and mean(sv) by position from left end
mean(abs(sv)) and mean(sv) by position from right end
mean(abs(sv)) and mean(sv) by aa type
mean(abs(sv)) and mean(sv) by aatype-posl
mean(abs(sv)) and mean(sv) by aatype-posr
mean(abs(sv)) and mean(sv) by 3-4 aa combo
"""

df = pd.read_pickle('results_dataframe.pkl')

# Bi token combos (1-based)
# For y ions
bc = np.array([
    [1,9],
    [8,9],
    [9,10],
    [1,10]
])

# mean(abs(sv)) and mean(sv) by position from left end
svsum_le = np.zeros((30))
svsum_re = np.zeros((30))
svabssum_le = np.zeros((30))
svabssum_re = np.zeros((30))
aa = {}
aapos_int_le = {}
aapos_le = {}
aapos_int_re = {}
aapos_re = {}
aa1 = {}
aaabs1 = {}
aa1int = {}
aa2 = {}
aaabs2 = {}
aa2int = {}
aa3 = {}
aaabs3 = {}
aa3int = {}
aa4 = {}
aaabs4 = {}
aa4int = {}
countsum = np.zeros((30))

for row in df.iterrows():
    l = len(row[1]['shap_values'])
    SV = np.array(row[1]['shap_values'])
    SQ = np.array(row[1]['sequence'])

    # Save total sums and counts by position
    count_le = np.append(np.ones((l)), np.zeros((30-l)))
    svsum_le[:l] += SV
    svsum_re[:l] += SV[::-1]
    svabssum_le[:l] += abs(SV)
    svabssum_re[:l] += abs(SV[::-1])
    countsum += count_le

    # Bi-token combo
    # Combo 1
    tok = '%c-%c'%(SQ[-bc[0,0]], SQ[-bc[0,1]])
    if tok not in aa1:
        aa1[tok] = []
    if tok not in aaabs1:
        aaabs1[tok] = []
    if tok not in aa1int:
        aa1int[tok] = []
    aa1[tok].append(SV[-bc[0,0]] + SV[-bc[0,1]])
    aaabs1[tok].append(abs(SV[-bc[0,0]]) + abs(SV[-bc[0,1]]))
    aa1int[tok].append(sum(SV))
    # Combo 2
    tok = '%c-%c'%(SQ[-bc[1,0]], SQ[-bc[1,1]])
    if tok not in aa2:
        aa2[tok] = []
    if tok not in aaabs2:
        aaabs2[tok] = []
    if tok not in aa2int:
        aa2int[tok] = []
    aa2[tok].append(SV[-bc[1,0]] + SV[-bc[1,1]])
    aaabs2[tok].append(abs(SV[-bc[1,0]]) + abs(SV[-bc[1,1]]))
    aa2int[tok].append(sum(SV))
    # Combo 3
    tok = '%c-%c'%(SQ[-bc[2,0]], SQ[-bc[2,1]])
    if tok not in aa3:
        aa3[tok] = []
    if tok not in aaabs3:
        aaabs3[tok] = []
    if tok not in aa3int:
        aa3int[tok] = []
    aa3[tok].append(SV[-bc[2,0]] + SV[-bc[2,1]])
    aaabs3[tok].append(abs(SV[-bc[2,0]]) + abs(SV[-bc[2,1]]))
    aa3int[tok].append(sum(SV))
    # Combo 4
    tok = '%c-%c'%(SQ[-bc[3,0]], SQ[-bc[3,1]])
    if tok not in aa4:
        aa4[tok] = []
    if tok not in aaabs4:
        aaabs4[tok] = []
    if tok not in aa4int:
        aa4int[tok] = []
    aa4[tok].append(SV[-bc[3,0]] + SV[-bc[3,1]])
    aaabs4[tok].append(abs(SV[-bc[3,0]]) + abs(SV[-bc[3,1]]))
    aa4int[tok].append(sum(SV))

    for i, (a,b) in enumerate(zip(SQ, SV)):

        # Store amino acid sv in list
        if a not in aa:
            aa[a] = []
        aa[a].append(b)
        
        # Define token as position from left end
        tok_le = '%c-%d'%(a, i)
        if tok_le not in aapos_le:
            aapos_le[tok_le] = []
        if tok_le not in aapos_int_le:
            aapos_int_le[tok_le] = []
        # Store values for token in list
        aapos_le[tok_le].append(b)
        aapos_int_le[tok_le].append(sum(SV))

        # Define token as position from right end
        tok_re = '%c-%d'%(a, l-i-1)
        if tok_re not in aapos_re:
            aapos_re[tok_re] = []
        if tok_re not in aapos_int_re:
            aapos_int_re[tok_re] = []
        # Store values for token in list
        aapos_re[tok_re].append(b)
        aapos_int_re[tok_re].append(sum(SV))

aasort = np.sort(list(aa.keys()))

min_occur = 300

svavg_le = svsum_le / (countsum + 1e-9)
svavg_le *= countsum>min_occur
svavg_re = svsum_re / (countsum + 1e-9)
svavg_re *= countsum>min_occur
svabsavg_le = svabssum_le / (countsum + 1e-9)
svabsavg_le *= countsum>min_occur
svabsavg_re = svabssum_re / (countsum + 1e-9)
svabsavg_re *= countsum>min_occur

aaabsavg = {a: np.mean(np.abs(aa[a])) for a in aasort}
aaavg = {a: np.mean(aa[a]) for a in aasort}
aastd = {a: np.std(aa[a]) for a in aasort}

min_occur = 15

# AA-position heatmaps
# Consolidate lists of values into single values for each token
aapos_int_le = {
    tok: np.mean(aapos_int_le[tok])
    for tok in aapos_int_le.keys()
    if len(aapos_int_le[tok])>min_occur
}
aapos_avg_le = {
    tok: np.mean(aapos_le[tok]) 
    for tok in aapos_le.keys()
    if len(aapos_le)>min_occur
}
aapos_absavg_le = {
    tok: np.mean(np.abs(aapos_le[tok])) 
    for tok in aapos_le.keys()
    if len(aapos_le)>min_occur
}
aapos_int_re = {
    tok: np.mean(aapos_int_re[tok])
    for tok in aapos_int_re.keys()
    if len(aapos_int_re[tok])>min_occur
}
aapos_avg_re = {
    tok: np.mean(aapos_re[tok]) 
    for tok in aapos_re.keys()
    if len(aapos_re[tok])>min_occur
}
aapos_absavg_re = {
    tok: np.mean(np.abs(aapos_re[tok])) 
    for tok in aapos_re.keys()
    if len(aapos_re[tok])>min_occur
}

# Order single values into aa x position heatmaps
heatmap_int_le = np.zeros((len(aasort), 30))
heatmap_le = np.zeros((len(aasort), 30))
heatmap_abs_le = np.zeros((len(aasort), 30))
heatmap_int_re = np.zeros((len(aasort), 30))
heatmap_re = np.zeros((len(aasort), 30))
heatmap_abs_re = np.zeros((len(aasort), 30))
for A,a in enumerate(aasort):
    for b in np.arange(30):
        tok = '%c-%d'%(a,b)
        if tok in aapos_int_le:
            heatmap_int_le[A,b] = aapos_int_le[tok]
        if tok in aapos_absavg_le:
            heatmap_le[A,b] = aapos_avg_le[tok]
            heatmap_abs_le[A,b] = aapos_absavg_le[tok]
        if tok in aapos_int_re:
            heatmap_int_re[A,b] = aapos_int_re[tok]
        if tok in aapos_absavg_re:
            heatmap_re[A,b] = aapos_avg_re[tok]
            heatmap_abs_re[A,b] = aapos_absavg_re[tok]

heatmap1 = np.zeros((len(aasort), len(aasort)))
heatmap1_abs = np.zeros((len(aasort), len(aasort)))
heatmap1_int = np.zeros((len(aasort), len(aasort)))
heatmap2 = np.zeros((len(aasort), len(aasort)))
heatmap2_abs = np.zeros((len(aasort), len(aasort)))
heatmap2_int = np.zeros((len(aasort), len(aasort)))
heatmap3 = np.zeros((len(aasort), len(aasort)))
heatmap3_abs = np.zeros((len(aasort), len(aasort)))
heatmap3_int = np.zeros((len(aasort), len(aasort)))
heatmap4 = np.zeros((len(aasort), len(aasort)))
heatmap4_abs = np.zeros((len(aasort), len(aasort)))
heatmap4_int = np.zeros((len(aasort), len(aasort)))
for A,a in enumerate(aasort):
    for B,b in enumerate(aasort):
        tok = '%c-%c'%(a,b)
        if tok in aa1:
            if len(aa1[tok])>min_occur:
                heatmap1[A,B] = np.mean(aa1[tok])
                heatmap1_abs[A,B] = np.mean(aaabs1[tok])
                heatmap1_int[A,B] = np.mean(aa1int[tok])
        if tok in aa2:
            if len(aa2[tok])>min_occur:
                heatmap2[A,B] = np.mean(aa2[tok])
                heatmap2_abs[A,B] = np.mean(aaabs2[tok])
                heatmap2_int[A,B] = np.mean(aa2int[tok])
        if tok in aa3:
            if len(aa3[tok])>min_occur:
                heatmap3[A,B] = np.mean(aa3[tok])
                heatmap3_abs[A,B] = np.mean(aaabs3[tok])
                heatmap3_int[A,B] = np.mean(aa3int[tok])
        if tok in aa4:
            if len(aa4[tok])>min_occur:
                heatmap4[A,B] = np.mean(aa4[tok])
                heatmap4_abs[A,B] = np.mean(aaabs4[tok])
                heatmap4_int[A,B] = np.mean(aa4int[tok])
