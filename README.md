Researching unsupervised and interpretable methods to find relationships between numerical and image-based data in multi-modal cancer data.

The main idea was to minimize the reconstruction error of two Autoencoders (one for numerical data, one for image-based data) while simultaneously maximizing the correlation (mean of the correlation matrix) between the (compressed) representations
of numerical and image-based data. Then we can calculate the correlation coefficients between the original numerical/image data and the compressed numerical/image data respectively, and by using these and the obtained correlation matrix mean, we 
introduce a "attention" weight function : for a specific sample, choose a specific numerical (rna) feature and patch ; then calculate the attention mapping for each of the image patch variables : specific_rna_feature_value * correlation_matrix_mean * curr_patch_value.
This creates a direct relationship between the image and numerical data, which can then be used to display the numerical data influence on the image data.
