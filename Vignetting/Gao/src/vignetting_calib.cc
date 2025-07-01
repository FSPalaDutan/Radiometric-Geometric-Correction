#define BACKWARD_HAS_DW 1
#include <backward.hpp>
namespace backward
{
backward::SignalHandling sh;
}

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <iomanip>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <camera_model/chessboard/Chessboard.h>
#include <camera_model/gpl/gpl.h>
#include <code_utils/cv_utils.h>
#include <vignetting_model/vignetting/vignettingcalib.h>
#include <vignetting_model/vignetting/vignettingtable.h>

int
main(int argc, char** argv)
{
    cv::Size boardSize;
    float squareSize;
    std::string inputDir;
    std::string cameraModel;
    std::string cameraName;
    std::string prefix;
    std::string fileExtension;
    bool useOpenCV;
    bool viewResults;
    bool verbose;
    bool is_save_images;
    std::string result_images_save_folder;
    std::string result_config_save_folder;
    int threshold = 60;
    float resize_scale = 1.0;
    cv::Size cropper_size(0, 0);
    cv::Point cropper_center(100, 100);
    bool is_color = false;

    // Handling Program Options
    using namespace boost::program_options;
    options_description desc("Allowed options.\n Ask GAO Wenliang if there is any possible questions.\n");
    desc.add_options()
        ("help", "produce help message")
        ("width,w", value<int>(&boardSize.width)->default_value(8), "Number of inner corners on the chessboard pattern in x direction")
        ("height,h", value<int>(&boardSize.height)->default_value(12), "Number of inner corners on the chessboard pattern in y direction")
        ("size,s", value<float>(&squareSize)->default_value(7.f), "Size of one square in mm")
        ("input,i", value<std::string>(&inputDir)->default_value("calibrationdata"), "Input directory containing chessboard images")
        ("prefix,p", value<std::string>(&prefix)->default_value(""), "Prefix of images")
        ("file-extension,e", value<std::string>(&fileExtension)->default_value(".png"), "File extension of images")
        ("camera-model", value<std::string>(&cameraModel)->default_value("mei"), "Camera model: kannala-brandt | fov | scaramuzza | mei | pinhole | myfisheye")
        ("camera-name", value<std::string>(&cameraName)->default_value("camera"), "Name of camera")
        ("opencv", value<bool>(&useOpenCV)->default_value(true), "Use OpenCV to detect corners")
        ("view-results", value<bool>(&viewResults)->default_value(true), "View results")
        ("verbose,v", value<bool>(&verbose)->default_value(true), "Verbose output")
        ("save_result", value<bool>(&is_save_images)->default_value(true), "save calibration result chessboard point.")
        ("result_images_save_folder", value<std::string>(&result_images_save_folder)->default_value("calib_images"), "calibration result images save folder.")
        ("result_config_save_folder", value<std::string>(&result_config_save_folder)->default_value(""), "Calibration result config save folder")
        ("resize-scale", value<float>(&resize_scale)->default_value(1.0f), "resize scale")
        ("cropper_width", value<int>(&cropper_size.width)->default_value(0), "cropper image width")
        ("cropper_height", value<int>(&cropper_size.height)->default_value(0), "cropper image height")
        ("center_x", value<int>(&cropper_center.x)->default_value(0), "cropper image center x")
        ("center_y", value<int>(&cropper_center.y)->default_value(0), "cropper image center y")
        ("is_color", value<bool>(&is_color)->default_value(false), "is_color")
        ("threshold", value<int>(&threshold)->default_value(60), "is_color");

    positional_options_description pdesc;
    pdesc.add("input", 1);

    variables_map vm;
    store(command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
    notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }

    if (!boost::filesystem::exists(inputDir) || !boost::filesystem::is_directory(inputDir))
    {
        std::cerr << "# ERROR: Cannot find input directory " << inputDir << "." << std::endl;
        return 1;
    }

    // Look for images in input directory
    std::vector<std::string> imageFilenames;
    boost::filesystem::directory_iterator itr;
    for (boost::filesystem::directory_iterator itr(inputDir); itr != boost::filesystem::directory_iterator(); ++itr)
    {
        if (!boost::filesystem::is_regular_file(itr->status()))
            continue;

        std::string filename = itr->path().filename().string();
        if (!prefix.empty() && filename.compare(0, prefix.length(), prefix) != 0)
            continue;
        if (filename.compare(filename.length() - fileExtension.length(), fileExtension.length(), fileExtension) != 0)
            continue;

        imageFilenames.push_back(itr->path().string());
        if (verbose)
            std::cerr << "# INFO: Adding " << imageFilenames.back() << std::endl;
    }

    if (imageFilenames.empty())
    {
        std::cerr << "# ERROR: No chessboard images found." << std::endl;
        return 1;
    }

    if (verbose)
        std::cerr << "# INFO: # images: " << imageFilenames.size() << std::endl;

    cv::Mat image = cv::imread(imageFilenames.front(), -1);
    cv::Size input_image_size(image.cols, image.rows);
    std::string file = cameraName + "_camera_calib.yaml";

    camera_model::VignettingCalib calib_vignetting(file, is_color);
    calib_vignetting.setBoardSize(boardSize);
    std::cout << " haha 3" << std::endl;

    std::vector<bool> chessboardFound(imageFilenames.size(), false);

    unsigned int index = 0;
    for (index = 0; index < imageFilenames.size(); ++index)
    {
        cv::Mat image_in = cv::imread(imageFilenames.at(index), -1);
        if (image_in.empty())
        {
            std::cerr << "# ERROR: Failed to load " << imageFilenames.at(index) << std::endl;
            continue;
        }

        cv::Mat image_for_corners;
        if (is_color && image_in.channels() == 3)
        {
            cv::cvtColor(image_in, image_for_corners, cv::COLOR_BGR2GRAY);
        }
        else
        {
            image_for_corners = image_in;
        }

        camera_model::Chessboard chessboard(boardSize, image_for_corners);
        chessboard.findCorners(useOpenCV);
        if (chessboard.cornersFound())
        {
            if (verbose)
                std::cerr << "# INFO: Detected chessboard in image " << index + 1 << ", " << imageFilenames.at(index) << std::endl;

            cv::Mat image_show = calib_vignetting.getChessboardPoints(image_in, chessboard.getCorners(), threshold);
            cv::Mat sketch;
            chessboard.getSketch().copyTo(sketch);

            if (viewResults)
            {
                cv::namedWindow("image_show", cv::WINDOW_NORMAL);
                cv::imshow("image_show", image_show);
                cv::waitKey(1);
            }
        }
        else if (verbose)
        {
            std::cout << "\033[31;47;1m" << "# INFO: Did not detect chessboard in image: " << imageFilenames.at(index) << "\033[0m" << std::endl;
        }
        chessboardFound.at(index) = chessboard.cornersFound();
    }
    // Condicionar la destrucción de la ventana "image_show"
    if (viewResults) {
        cv::destroyWindow("image_show");
    }
    std::cout << " haha 4" << std::endl;

    if (verbose)
        std::cerr << "# INFO: Calibrating..." << std::endl;

    double startTime = camera_model::timeInSeconds();
    std::cout << " Calibrate start." << std::endl;
    calib_vignetting.solve();
    std::cout << " Calibrate done." << std::endl;

    if (verbose)
        std::cout << "# INFO: Calibration took a total time of " << std::fixed << std::setprecision(3)
                  << camera_model::timeInSeconds() - startTime << " sec.\n";

    // Modificación: Pasar result_images_save_folder a showResualt
    boost::filesystem::create_directories(result_images_save_folder); // Asegurar que la carpeta exista
    calib_vignetting.showResualt(result_images_save_folder);

    std::string config_save_path = result_config_save_folder.empty() ? inputDir : result_config_save_folder;
    if (is_save_images && !config_save_path.empty())
    {
        boost::filesystem::create_directories(config_save_path);
        if (verbose)
            std::cerr << "# INFO: Config files will be saved to " << config_save_path << std::endl;
    }
    calib_vignetting.writeToYamlFile(config_save_path + "/" + cameraName + "_vignetting_calib.yaml");

    camera_model::VignettingTable vignetting(cameraName + "_vignetting_calib.yaml");

    for (size_t i = 0; i < imageFilenames.size(); ++i)
    {
        cv::Mat image_in = cv::imread(imageFilenames.at(i), -1);
        if (image_in.empty())
        {
            std::cerr << "# ERROR: Failed to load " << imageFilenames.at(i) << std::endl;
            continue;
        }

        startTime = camera_model::timeInSeconds();
        cv::Mat image_remove = calib_vignetting.remove(image_in);
        std::cout << "# INFO: Remove cost " << std::fixed << std::setprecision(3)
                  << (camera_model::timeInSeconds() - startTime) * 1000 << " ms.\n";

        startTime = camera_model::timeInSeconds();
        cv::Mat image_lut = vignetting.removeLUT(image_in);
        std::cout << "# INFO: removeLUT cost " << std::fixed << std::setprecision(3)
                  << (camera_model::timeInSeconds() - startTime) * 1000 << " ms.\n";

        if (viewResults)
        {
            cv::namedWindow("image_in", cv::WINDOW_NORMAL);
            cv::namedWindow("image_remove", cv::WINDOW_NORMAL);
            cv::namedWindow("image_lut", cv::WINDOW_NORMAL);
            cv::imshow("image_in", image_in);
            cv::imshow("image_remove", image_remove);
            cv::imshow("image_lut", image_lut);
            cv::waitKey(0);
        }

        if (is_save_images && !result_images_save_folder.empty())
        {
            boost::filesystem::create_directories(result_images_save_folder);
            std::string filename = boost::filesystem::path(imageFilenames.at(i)).filename().string();
            
            // Guardar imagen corregida por removeLUT
            std::string lut_save_path = result_images_save_folder + "/corrected_LUT_" + filename;
            std::cout << "Guardando imagen de removeLUT en: " << lut_save_path << " con dimensiones: " 
                      << image_lut.cols << "x" << image_lut.rows << std::endl;
            cv::imwrite(lut_save_path, image_lut);
            if (verbose)
                std::cerr << "# INFO: Saved LUT-corrected image to " << lut_save_path << std::endl;

            // Guardar imagen corregida por remove
            std::string remove_save_path = result_images_save_folder + "/corrected_remove_" + filename;
            std::cout << "Guardando imagen de remove en: " << remove_save_path << " con dimensiones: " 
                      << image_remove.cols << "x" << image_remove.rows << std::endl;
            cv::imwrite(remove_save_path, image_remove);
            if (verbose)
                std::cerr << "# INFO: Saved remove-corrected image to " << remove_save_path << std::endl;
        }
    }

    // Cerrar las ventanas de la segunda sección si viewResults es true
    if (viewResults) {
        cv::destroyWindow("image_in");
        cv::destroyWindow("image_remove");
        cv::destroyWindow("image_lut");
    }

    return 0;
}
