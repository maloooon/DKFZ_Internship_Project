import pandas as pd 



data = pd.read_csv('Slides/data/tcga_luad_all_clean.csv')


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
        # feature-wise normalizing
        rna_df[columnName] = (columnData - columnData.mean()) / columnData.std()
        counter += 1
        if counter > 3000:
            pass
    else:
        other_df[columnName] = columnData





cnv_df.to_csv('Slides/data/cnv.csv')
rna_df.to_csv('Slides/data/rnaseq.csv')
other_df.to_csv('Slides/data/other.csv')

