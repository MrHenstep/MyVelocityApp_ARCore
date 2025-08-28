import B1_build_model_depth_batch as gen
import L1_lib_extraction_and_visualisation as exv


if __name__ == "__main__":
        
    ##########################################################################################################

    # FILE_PATH = "c:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported"

    # FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_27_drive_full_pipeline_test"
    FILE_PATH = "C:\\Users\\steph\\Documents\\Projects\\AndroidStudioProjects\\Velociraptor-app\\exported\\2025_08_27_static_test"


    CROP_NOT_SCALE = True

    # ROTATE_K = 3 # if the phone is vertical
    ROTATE_K = 0 # if the phone is horizontal

    MIDAS_MODEL_WEIGHTS = {
        "midas_v21_small": r"C:\Users\steph\Documents\Projects\AndroidStudioProjects\Velociraptor-app\MiDaS_weights\midas_v21_small-70d6b9c8.pt",
        "midas_v21": r"C:\Users\steph\Documents\Projects\AndroidStudioProjects\Velociraptor-app\MiDaS_weights\midas_v21-f6b98070.pt",
        "dpt_large": r"C:\Users\steph\Documents\Projects\AndroidStudioProjects\Velociraptor-app\MiDaS_weights\dpt_large-midas-2f21e586.pt",
        "dpt_beit_large_512": r"C:\Users\steph\Documents\Projects\AndroidStudioProjects\Velociraptor-app\MiDaS_weights\dpt_beit_large_512.pt"
        # Add more models and their weights paths here as needed
        
    }

    ##########################################################################################################

    # model_name_list = ["midas_v21", "dpt_large", "dpt_beit_large_512", "depth_anything_v2_small", "depth_anything_v2_base", "depth_anything_v2_large"]
    model_name_list = ["depth_anything_v2_large"]

    ###############################################################################################################

    BATCH_NUMBER_LIST = [0, 1, 2, 3]
    # BATCH_NUMBER_LIST = [0]

    for batch_number in BATCH_NUMBER_LIST:

        MATCHED_INDICES = exv.get_all_indices(FILE_PATH, batch_number)
        MATCHED_FILENAME_TABLE = exv.get_matched_filenames(MATCHED_INDICES, FILE_PATH, batch_number)

        for index, row in enumerate(MATCHED_FILENAME_TABLE):

            # if (index != 0): continue

            file_name = row[1]  # camera file

            depth_grey_phone_original = gen.load_rgb_from_float6(FILE_PATH, file_name.replace("camera", "grey"), 480, 640)
            depth_grey_phone = gen.rot90k(depth_grey_phone_original, ROTATE_K)

            rgb_original = gen.load_rgb_from_float6(FILE_PATH, file_name, 480, 640)
            rgb = gen.rot90k(rgb_original, ROTATE_K)
            image_height, image_width = rgb.shape[:2]


            saved_images = []
            saved_images.append((file_name, rgb, None))
            saved_images.append((file_name.replace("camera", "grey"), depth_grey_phone, None))

            for model_name in model_name_list:
                
                print(f"Displaying {model_name} on image file {file_name} ...")

                depth_on_full = gen.load_rgb_from_float6(FILE_PATH, file_name.replace("camera", "MOD_"+model_name), 480, 640)
                saved_images.append((model_name, depth_on_full, None))

                print ("\n")

            gen.show_rgb_images(*[img for _, img, _ in saved_images], titles=[name for name, _, _ in saved_images])
