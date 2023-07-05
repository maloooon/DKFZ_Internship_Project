import pandas as pd 



data = pd.read_csv('Slides/data/tcga_luad_all_clean.csv')
#data = pd.read_csv('Slides/data/TCGA_LUAD_1_mRNA.csv')


# cnv, own names, rnaseq

cnv_df = pd.DataFrame()
rna_df = pd.DataFrame()
other_df = pd.DataFrame()

# TESTING COUNT ; just take 3000 RNA feats
counter = 0

for (columnName, columnData) in data.iteritems():
    if 'cnv' in columnName:
        # feature-wise normalizing
        cnv_df[columnName] = (columnData - columnData.mean()) / columnData.std()
    elif 'rnaseq' in columnName:
          
        if counter >= 1000:
            pass
        else:  
            # feature-wise normalizing  
           # rna_df[columnName] = (columnData - columnData.mean()) / columnData.std()
            rna_df[columnName] = (columnData - columnData.min()) / (columnData.max() - columnData.min())
          #   global normalizing (over first 1000)
          #  rna_df[columnName] = columnData
        counter += 1
    
    elif 'mRNA' in columnName:
        if counter >= 1000:
            pass
        else:
            rna_df[columnName] = (columnData - columnData.min()) / (columnData.max() - columnData.min())
        counter += 1


    else:
        other_df[columnName] = columnData


# global normalizing
#rna_std_global = rna_df.stack().std()
#rna_mean_global = rna_df.stack().mean()

#for column in rna_df:
#    rna_df[column] = (rna_df[column] - rna_mean_global) / rna_std_global

rna_df = rna_df.fillna(0)

#rna_df = rna_df.iloc[0:500] # TEST PURPOSES
cnv_df.to_csv('Slides/data/cnv.csv')
rna_df.to_csv('Slides/data/rnaseq.csv')
other_df.to_csv('Slides/data/other.csv')

