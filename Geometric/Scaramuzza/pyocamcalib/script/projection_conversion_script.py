from typing import Tuple
import typer
import cv2 as cv
import matplotlib.pyplot as plt
import os

from pyocamcalib.modelling.camera import Camera


def main(fisheye_image_path: str,
         calibration_file_path: str,
         perspective_fov: float,
         perspective_sensor_size: Tuple[int, int],
         ):
    """
    :param fisheye_image_path: fisheye image path.
    :param calibration_file_path: .json file with calibration parameters.
    :param perspective_fov: field of view the desired perspective camera in degree (between 0 and 180).
    :param perspective_sensor_size: (height, width) in pixels. Provided for compatibility, but overridden by original image dimensions.
    :return:
    """

    # Load the fisheye image
    fisheye_image = cv.imread(fisheye_image_path)
    if fisheye_image is None:
        raise ValueError(f"Could not load image at {fisheye_image_path}")

    # Get the original image dimensions (height, width)
    height, width = fisheye_image.shape[:2]
    perspective_sensor_size = (height, width)  # Override with original dimensions

    # Load calibration parameters and perform perspective conversion
    my_camera = Camera.load_parameters_json(calibration_file_path)
    perspective_image = my_camera.cam2perspective_indirect(fisheye_image,
                                                           perspective_fov,
                                                           perspective_sensor_size)

    # Save the corrected image
    directory = os.path.dirname(fisheye_image_path)
    corregidas_dir = os.path.join(directory, "corregidas")
    if not os.path.exists(corregidas_dir):
        os.makedirs(corregidas_dir)
    base_filename = os.path.basename(fisheye_image_path)
    new_filename = "corregida_" + base_filename
    full_path = os.path.join(corregidas_dir, new_filename)
    cv.imwrite(full_path, perspective_image)
    print(f"Imagen corregida guardada en {full_path}")

    # Display images (optional, can be removed if not needed)



if __name__ == "__main__":
    typer.run(main)