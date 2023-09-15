import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

scaler = MinMaxScaler()



########## OLD DATA ##########
"""
data_mrna = pd.read_csv('Slides/data/tests/TCGA_PRAD_1_mrna.csv',index_col=0)
data_dna = pd.read_csv('Slides/data/tests/TCGA_PRAD_2_dna.csv',index_col=0)



#data_mrna = data_mrna.iloc[:,1:1000]
data_mrna = scaler.fit_transform(data_mrna.iloc[:, 1:1000])
data_dna = scaler.fit_transform(data_dna.iloc[:, 1:1000])


mrna_df = pd.DataFrame(data_mrna)
dna_df = pd.DataFrame(data_dna)
mrna_df.fillna(0,inplace=True)
dna_df.fillna(0,inplace=True)

mrna_df.to_csv('Slides/data/tests/mRNA.csv')
dna_df.to_csv('Slides/data/tests/DNA.csv')
"""

########## OLD DATA ##########



data = pd.read_csv('Slides/data/tcga_luad_all_clean.csv')


# cnv, own names, rnaseq

cnv_df = pd.DataFrame()
rna_df = pd.DataFrame()
other_df = pd.DataFrame()

# TESTING COUNT ; just take 1000 RNA feats
counter = 0

for (columnName, columnData) in data.iteritems():
    if 'cnv' in columnName:
        # feature-wise normalizing
        cnv_df[columnName] = columnData 
    elif 'rnaseq' in columnName: 
     #   if counter >= 1000:
     #       pass
      #  else:
        rna_df[columnName] = columnData 
     #   counter += 1

    else:
        other_df[columnName] = columnData


rna_numpy= scaler.fit_transform(rna_df)

threshold = 0

while rna_numpy.shape[1] > 2000:
    variance_selection = VarianceThreshold(threshold= threshold)
    rna_numpy = variance_selection.fit_transform(rna_numpy)
    threshold += 0.01


print(rna_numpy.shape[1])
rna_df = pd.DataFrame(rna_numpy).fillna(0)


cnv_df.to_csv('Slides/data/cnv.csv')
rna_df.to_csv('Slides/data/rnaseq.csv')
other_df.to_csv('Slides/data/other.csv')