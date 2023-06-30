import wsi_preprocessing as pp

if __name__ == '__main__':
    # if, for instance, CMU-1.svs is in your current directory ("."):
    sample_nr = 3
    slides = pp.list_slides("Slides/slide_data/sample{}/".format(sample_nr))
    pp.save_slides_mpp_otsu(slides, "Slides/slide_data/sample{}/slides_mpp_otsu.csv".format(sample_nr))

    pp.run_tiling("Slides/slide_data/sample{}/slides_mpp_otsu.csv".format(sample_nr), "Slides/slide_data/sample{}/tiles.csv".format(sample_nr))

    pp.calculate_filters("Slides/slide_data/sample{}/slides_mpp_otsu.csv".format(sample_nr),"", "Slides/slide_data/sample{}/tiles_filters.csv".format(sample_nr))