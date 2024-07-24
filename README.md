The main idea was to minimize the reconstruction error of two Autoencoders (one for numerical data, one for image-based data) while simultaneously maximizing the correlation (mean of the correlation matrix) between the (compressed) representations
of numerical and image-based data. Then we can calculate the correlation coefficients between the original numerical/image data and the compressed numerical/image data respectively, and by using these and the obtained correlation matrix mean, we 
introduce a "attention" weight function : for a specific sample, choose a specific numerical (rna) feature and patch ; then calculate the attention mapping for each of the image patch variables : specific_rna_feature_value * correlation_matrix_mean * curr_patch_value.
This creates a direct relationship between the image and numerical data. By then utilizing the deocder structure of imgAE, we can depict the effect of the RNA feature on different image patches. 



Results on an examplary RNA feature on 3 image patches. The top row shows the original image patch and the bottom row the heatmap of the effect of the RNA feature on it.
![alt text](https://github.com/maloooon/CCA-Project/blob/main/results_example.png?raw=true)

