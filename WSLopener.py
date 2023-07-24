import wsi_preprocessing as pp

if __name__ == '__main__':
    # if, for instance, CMU-1.svs is in your current directory ("."):

    for i in range(17,20):
        try:
            sample_nr = i

            slides = pp.list_slides("0to128/{}".format(sample_nr))
            pp.save_slides_mpp_otsu(slides, "0to128/{}/slides_mpp_otsu.csv".format(sample_nr))
            pp.run_tiling("0to128/{}/slides_mpp_otsu.csv".format(sample_nr), "0to128/{}/tiles.csv".format(sample_nr))
        except Exception as e:
            pass

   # slides = pp.list_slides("Slides/slide_data/sample_new_{}/".format(sample_nr))
   # pp.save_slides_mpp_otsu(slides, "Slides/slide_data/sample_new_{}/slides_mpp_otsu.csv".format(sample_nr))

   # pp.run_tiling("Slides/slide_data/sample_new_{}/slides_mpp_otsu.csv".format(sample_nr), "Slides/slide_data/sample_new_{}/tiles.csv".format(sample_nr))
