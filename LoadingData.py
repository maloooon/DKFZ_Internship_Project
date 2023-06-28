import pandas as pd 



data = pd.read_csv('Slides/data/tcga_luad_all_clean.csv')


# cnv, own names, rnaseq

cnv_df = pd.DataFrame()
rna_df = pd.DataFrame()
other_df = pd.DataFrame()

for (columnName, columnData) in data.iteritems():
    if 'cnv' in columnName:
        cnv_df[columnName] = columnData
    elif 'rnaseq' in columnName:
        rna_df[columnName] = columnData
    else:
        other_df[columnName] = columnData

cnv_df.to_csv('Slides/data/cnv.csv')
rna_df.to_csv('Slides/data/rnaseq.csv')
other_df.to_csv('Slides/data/other.csv')

