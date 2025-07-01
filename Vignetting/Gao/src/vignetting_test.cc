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
#include <vignetting_model/vignetting/vignettingtable.h>

int main(int argc, char** argv)
{
    std::cout << "=== INICIO DEL PROGRAMA ===" << std::endl;
    
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

    float resize_scale = 1.0;
    cv::Size cropper_size(0, 0);
    cv::Point cropper_center(100, 100);
    bool is_color = false;

    //========= Handling Program options =========
    std::cout << "=== CONFIGURANDO OPCIONES ===" << std::endl;
    using namespace boost::program_options;
    boost::program_options::options_description desc(
    "Allowed options.\n Ask GAO Wenliang if there is any possible questions.\n");
    desc.add_options()
        ("help", "produce help message")
        ("width,w", value<int>(&boardSize.width)->default_value(8), "Number of inner corners on the chessboard pattern in x direction")
        ("height,h", value<int>(&boardSize.height)->default_value(12), "Number of inner corners on the chessboard pattern in y direction")
        ("size,s", value<float>(&squareSize)->default_value(7.f), "Size of one square in mm")
        ("input,i", value<std::string>(&inputDir)->default_value("calibrationdata"), "Input directory containing chessboard images")
        ("prefix,p", value<std::string>(&prefix)->default_value(""), "Prefix of images")
        ("file-extension,e", value<std::string>(&fileExtension)->default_value(".png"),"File extension of images")
        ("camera-model", value<std::string>(&cameraModel)->default_value("mei"),"Camera model: kannala-brandt | fov | scaramuzza | mei | pinhole | myfisheye")
        ("camera-name", value<std::string>(&cameraName)->default_value("camera"), "Name of camera")
        ("opencv", value<bool>(&useOpenCV)->default_value(true), "Use OpenCV to detect corners")
        ("view-results", value<bool>(&viewResults)->default_value(true), "View results")
        ("verbose,v", value<bool>(&verbose)->default_value(true), "Verbose output")
        ("save_result", value<bool>(&is_save_images)->default_value(true), "save calibration result chessboard point.")
        ("result_images_save_folder", value<std::string>(&result_images_save_folder)->default_value("corregidas_test"), "calibration result images save folder.")
        ("resize-scale", value<float>(&resize_scale)->default_value(1.0f), "resize scale")
        ("cropper_width", value<int>(&cropper_size.width)->default_value(0), "cropper image width")
        ("cropper_height", value<int>(&cropper_size.height)->default_value(0), "cropper image height")
        ("center_x", value<int>(&cropper_center.x)->default_value(0), "cropper image center x ")
        ("center_y", value<int>(&cropper_center.y)->default_value(0), "cropper image center y ")
        ("is_color", value<bool>(&is_color)->default_value(false), "is_color ")
        ;

    boost::program_options::positional_options_description pdesc;
    pdesc.add("input", 1);

    boost::program_options::variables_map vm;
    boost::program_options::store(
    boost::program_options::command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
    boost::program_options::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }

    std::cout << "=== VERIFICANDO DIRECTORIO DE ENTRADA ===" << std::endl;
    if (!boost::filesystem::exists(inputDir) || !boost::filesystem::is_directory(inputDir))
    {
        std::cerr << "# ERROR: No se puede encontrar o no es un directorio de entrada válido: " << inputDir << "." << std::endl;
        return 1;
    }

    // Buscar imágenes en el directorio de entrada
    std::cout << "=== BUSCANDO IMAGENES EN EL DIRECTORIO ===" << std::endl;
    std::vector<std::string> imageFilenames;
    for (boost::filesystem::directory_iterator itr(inputDir); itr != boost::filesystem::directory_iterator(); ++itr)
    {
        if (!boost::filesystem::is_regular_file(itr->status()))
        {
            continue;
        }

        std::string filename = itr->path().filename().string();

        // Verificar si el prefijo coincide
        if (!prefix.empty() && filename.compare(0, prefix.length(), prefix) != 0)
        {
            continue;
        }

        // Verificar si la extensión coincide
        if (filename.compare(filename.length() - fileExtension.length(), fileExtension.length(), fileExtension) != 0)
        {
            continue;
        }

        imageFilenames.push_back(itr->path().string());

        if (verbose)
        {
            std::cerr << "# INFO: Añadiendo " << imageFilenames.back() << std::endl;
        }
    }

    if (imageFilenames.empty())
    {
        std::cerr << "# ERROR: No se encontraron imágenes de tablero de ajedrez." << std::endl;
        return 1;
    }

    if (verbose)
    {
        std::cerr << "# INFO: # imágenes: " << imageFilenames.size() << std::endl;
    }

    std::cout << "=== CARGANDO PRIMERA IMAGEN PARA OBTENER TAMAÑO ===" << std::endl;
    cv::Mat image = cv::imread(imageFilenames.front(), -1);
    if (image.empty())
    {
        std::cerr << "# ERROR: No se pudo cargar la primera imagen: " << imageFilenames.front() << std::endl;
        return 1;
    }
    
    cv::Size input_image_size(image.cols, image.rows);
    std::string file = cameraName + "_camera_calib.yaml";

    std::cout << "=== PUNTO DE CONTROL 1 ===" << std::endl;
    std::cout << "haha 3" << std::endl;

    std::vector<bool> chessboardFound(imageFilenames.size(), false);

    double startTime = camera_model::timeInSeconds();

    std::cout << "=== INTENTANDO CARGAR MODELO DE VIÑETEO ===" << std::endl;
    std::string vignettingFile = cameraName + "_vignetting_calib.yaml";
    std::cout << "Buscando archivo: " << vignettingFile << std::endl;
    
    if (!boost::filesystem::exists(vignettingFile))
    {
        std::cerr << "# ERROR: Archivo de viñeteo no encontrado: " << vignettingFile << std::endl;
        std::cerr << "Directorio actual: " << boost::filesystem::current_path() << std::endl;
        return 1;
    }

    std::cout << "Archivo encontrado. Tamaño: " << boost::filesystem::file_size(vignettingFile) << " bytes" << std::endl;
    
    try
    {
        std::cout << "Creando VignettingTable..." << std::endl;
        camera_model::VignettingTable vignettingTable(vignettingFile);
        std::cout << "VignettingTable creado exitosamente" << std::endl;
        
        std::cout << "Creando objeto vignetting..." << std::endl;
        camera_model::vignetting vignetting(vignettingFile);
        std::cout << "Objeto vignetting creado exitosamente" << std::endl;

        std::cout << "Mostrando resultados..." << std::endl;
	vignetting.showResualt("."); // Ahora guardará en vez de mostrar
        std::cout << "Resultados mostrados" << std::endl;

        // Crear la carpeta de destino si no existe
        if (is_save_images && !result_images_save_folder.empty())
        {
            std::cout << "Creando directorio para guardar resultados: " << result_images_save_folder << std::endl;
            boost::filesystem::create_directories(result_images_save_folder);
        }

        std::cout << "=== PROCESANDO IMAGENES ===" << std::endl;
        for (size_t i = 0; i < imageFilenames.size(); ++i)
        {
            std::cout << "Procesando imagen " << i + 1 << "/" << imageFilenames.size() << ": " << imageFilenames.at(i) << std::endl;
            
            cv::Mat image_in = cv::imread(imageFilenames.at(i), -1);
            if (image_in.empty())
            {
                std::cerr << "# ERROR: No se pudo cargar la imagen: " << imageFilenames.at(i) << std::endl;
                continue;
            }

            std::cout << "  Eliminando viñeteo (método directo)..." << std::endl;
            startTime = camera_model::timeInSeconds();
            cv::Mat image_show = vignetting.remove(image_in);
            std::cout << "# INFO: Tiempo de corrección: " << std::fixed << std::setprecision(3)
                      << (camera_model::timeInSeconds() - startTime) * 1000 << " ms.\n";

            // Guardar solo la imagen corregida si se solicita
            if (is_save_images && !result_images_save_folder.empty())
            {
                std::string filename = boost::filesystem::path(imageFilenames.at(i)).filename().string();
                std::string save_path = result_images_save_folder + "/corrected_" + filename;
                std::cout << "  Guardando imagen corregida en: " << save_path << std::endl;
                cv::imwrite(save_path, image_show); // Guardamos solo la imagen corregida
                if (verbose)
                {
                    std::cerr << "# INFO: Imagen corregida guardada en " << save_path << std::endl;
                }
            }

            // Mostrar imágenes si viewResults es true (esto no afecta el guardado)
            if (viewResults)
            {
                std::cout << "  Mostrando resultados..." << std::endl;
                cv::Mat image_out = vignettingTable.removeLUT(image_in); // Calcular LUT solo para visualización
                cv::namedWindow("image_in", cv::WINDOW_NORMAL);
                cv::namedWindow("image_out", cv::WINDOW_NORMAL);
                cv::namedWindow("image_LUT", cv::WINDOW_NORMAL);
                cv::imshow("image_in", image_in);
                cv::imshow("image_out", image_show);
                cv::imshow("image_LUT", image_out);
                cv::waitKey(0);
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "# EXCEPCIÓN: " << e.what() << std::endl;
        std::cerr << "# ERROR: No se pudo cargar el modelo de viñeteo correctamente" << std::endl;
        return 1;
    }

    std::cout << "=== PROGRAMA COMPLETADO EXITOSAMENTE ===" << std::endl;
    return 0;
}
