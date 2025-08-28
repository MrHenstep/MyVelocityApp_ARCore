import matplotlib.pyplot as plt

import L2_lib_batch_fit_regression as bfr
import L1_lib_extraction_and_visualisation as exv

import persistence_helpers as phelp

###########################################################################################################

if __name__ == "__main__":

    ##########################################################################################################

    # FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"
    FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_27_static_test"


    ###########################################################################################################


    CONFIDENCE_LEVEL = 0.75
    DEPTH_MAX = 25.0
    DEPTH_RANGE_FOR_COLOUR_MAP = (0.0, DEPTH_MAX)

    FRAME_INCLUSION_LIST = range(0, 100, 1)

    MATCH_TIMESTAMPS = False

    # Apply a sigmoid weight dropping towards zero above 7, width 1
    X_CUT = 7.0
    X_WIDTH = 1.0
    WEIGHTS_SIGMOID = (X_CUT, X_WIDTH)

    INVERT_AXES = False
    DISPLAY_PLOTS = True

    WIDTH = 640
    HEIGHT = 480

    ###########################################################################################################


    BATCH_NUMBER_LIST = [0, 1, 2, 3]

    for batch_number in BATCH_NUMBER_LIST:
        
        TIMESTAMPS_TABLE = exv.read_timestamp_files(FILE_PATH, batch_number)

        if MATCH_TIMESTAMPS:
            MATCHED_INDICES = exv.find_closest_timestamp_matches(TIMESTAMPS_TABLE, 3, 2, direction='both')
        else:
            MATCHED_INDICES = exv.get_all_indices(FILE_PATH, batch_number)

        MATCHED_FILENAME_TABLE = exv.get_matched_filenames(MATCHED_INDICES, FILE_PATH, batch_number)


        # depth_model_name_list = ["Phone_midas_v21_small", "midas_v21" , "dpt_large", "dpt_beit_large_512", "depth_anything_v2_small", "depth_anything_v2_base", "depth_anything_v2_large"]
        depth_model_name_list = ["Phone_midas_v21_small", "depth_anything_v2_large"]
        regression_results_list = []
        regression_results_dict = {}
        invert_axes_list = []
        depth_max_list = []

        for depth_model_name in depth_model_name_list:

            if "Phone" in depth_model_name:
                depth_map_file_name_replacement = None
            else:
                depth_map_file_name_replacement = "MOD_"+depth_model_name

            regression = bfr.batch_fit_regression_model(
                FILE_PATH, 
                batch_number, 
                FRAME_INCLUSION_LIST, 
                depth_map_file_name_replacement, 
                confidence_level=CONFIDENCE_LEVEL, 
                depth_range_for_colour_map=DEPTH_RANGE_FOR_COLOUR_MAP, 
                weights_sigmoid=WEIGHTS_SIGMOID, 
                display_plots=DISPLAY_PLOTS, 
                match_timestamps=MATCH_TIMESTAMPS
            )

            bfr.plot_regression_fits(regression, INVERT_AXES, DEPTH_MAX, depth_model_name)
            plt.tight_layout()
            plt.show()

            regression_results_list.append(regression)
            invert_axes_list.append(INVERT_AXES)
            depth_max_list.append(DEPTH_MAX)

            regression_results_dict[depth_model_name] = regression

        bfr.plot_multiple_regression_fits(
            regression_results_list,
            invert_axes_list,
            depth_max_list,
            depth_model_name_list
        )

        phelp.save_dataframe_dict_to_csv(regression_results_dict, FILE_PATH + "\\" + f"batch_{batch_number}_regression_results_dict.csv")    

