#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <sys/stat.h> // Para mkdir
#include <sys/types.h>

#include "io.h"
#include "ocv.h"
#include "eigen.h"
#include "json.h"

#include "geometry/geometry.h"
#include "projection/pinhole.h"
#include "projection/eucm.h"
#include "projection/ucm.h"
#include "projection/mei.h"

using namespace std;
using namespace cv;

// Función auxiliar para extraer el nombre base de un archivo (sin ruta ni extensión)
string getBaseName(const string& path) {
    // Encontrar la última barra (separador de directorio)
    size_t last_slash = path.find_last_of("/\\");
    string filename = (last_slash == string::npos) ? path : path.substr(last_slash + 1);
    
    // Encontrar la última extensión
    size_t last_dot = filename.find_last_of(".");
    if (last_dot == string::npos) {
        return filename;
    }
    return filename.substr(0, last_dot);
}

// Función auxiliar para combinar directorio y nombre de archivo
string joinPath(const string& dir, const string& filename) {
    if (dir.empty() || dir == "./") {
        return filename;
    }
    // Asegurar que el directorio termine con una barra
    string clean_dir = dir;
    if (clean_dir.back() != '/' && clean_dir.back() != '\\') {
        clean_dir += '/';
    }
    return clean_dir + filename;
}

// Función auxiliar para crear directorio (POSIX)
bool createDirectory(const string& dir) {
    if (dir.empty() || dir == "./") {
        return true;
    }
    // Crear directorio con permisos 0755
    int status = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    return (status == 0 || errno == EEXIST);
}

void initRemap(const vector<double>& params, const vector<double>& pinhole_params,
               Mat32f & mapX, Mat32f & mapY, const Transf & T)
{
    Pinhole cam2(pinhole_params[2], pinhole_params[3], pinhole_params[4]);

    Vector2dVec imagePoints;
    mapX.create(pinhole_params[1], pinhole_params[0]);
    mapY.create(pinhole_params[1], pinhole_params[0]);
    for (unsigned int i = 0; i < mapX.rows; i++) {
        for (unsigned int j = 0; j < mapX.cols; j++) {
            imagePoints.push_back(Vector2d(j, i));
        }
    }

    Vector3dVec pointCloud;
    cam2.reconstructPointCloud(imagePoints, pointCloud);

    T.transform(pointCloud, pointCloud);

    if(params.size() == 6)
    {
        EnhancedCamera cam(params.data());
        cam.projectPointCloud(pointCloud, imagePoints);
    }
    else if(params.size() == 5)
    {
        UnifiedCamera cam(params.data());
        cam.projectPointCloud(pointCloud, imagePoints);
    }
    else if(params.size() == 10)
    {
        MeiCamera cam(params.data());
        cam.projectPointCloud(pointCloud, imagePoints);
    }
    else
    {
        cerr << "Parámetros de cámara no soportados." << endl;
        exit(1);
    }

    auto pointIter = imagePoints.begin();
    for (unsigned int i = 0; i < mapX.rows; i++) {
        for (unsigned int j = 0; j < mapX.cols; j++) {
            mapX(i, j) = (*pointIter)[0];
            mapY(i, j) = (*pointIter)[1];
            ++pointIter;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Uso: " << argv[0] << " <archivo_json> [directorio_salida]" << endl;
        return 1;
    }

    string json_file = argv[1];
    string output_dir = (argc >= 3) ? argv[2] : "./corregidas";

    // Crear el directorio de salida si no existe
    if (!createDirectory(output_dir)) {
        cerr << "Error: No se pudo crear el directorio " << output_dir << endl;
        return 1;
    }

    ptree root;
    read_json(json_file, root);
    vector<double> paramsCam = readVector<double>(root.get_child("camera_params"));
    vector<double> paramsPinhole = readVector<double>(root.get_child("pinhole_params"));
    Transf T = readTransform(root.get_child("xi_eucm_pinhole"));

    Mat32f mapX, mapY;
    initRemap(paramsCam, paramsPinhole, mapX, mapY, T);

    for (auto & x : root.get_child("image_names")) {
        string input_path = x.second.get_value<string>();
        Mat img = imread(input_path, IMREAD_COLOR);
        if (img.empty()) {
            cerr << "Error: No se pudo leer la imagen " << input_path << endl;
            continue;
        }

        Mat img_float, img2;
        img.convertTo(img_float, CV_32FC3);
        remap(img_float, img2, mapX, mapY, INTER_LINEAR);

        Mat img_output;
        img2.convertTo(img_output, CV_8UC3);

        string base_name = getBaseName(input_path);
        string output_filename = "corregida_" + base_name + ".jpg";
        string output_path = joinPath(output_dir, output_filename);

        if (imwrite(output_path, img_output)) {
            cout << "Imagen guardada en: " << output_path << endl;
        } else {
            cerr << "Error al guardar la imagen en: " << output_path << endl;
        }
    }

    return 0;
}