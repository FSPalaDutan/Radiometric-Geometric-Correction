#include <iostream>
#include <vector>
#include <cstdlib>

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

void initRemap(const vector<double>& params, const vector<double>& pinhole_params,
               Mat32f & mapX, Mat32f & mapY, const Transf & T)
{
    // Crear cámara pinhole
    Pinhole cam2(pinhole_params[2], pinhole_params[3], pinhole_params[4]);

    mapX.create(pinhole_params[1], pinhole_params[0]); // rows, cols
    mapY.create(pinhole_params[1], pinhole_params[0]);

    Vector2dVec imagePoints;
    for (int i = 0; i < mapX.rows; ++i) {
        for (int j = 0; j < mapX.cols; ++j) {
            imagePoints.emplace_back(j, i); // (u, v)
        }
    }

    // Obtener nube de puntos (rayos)
    Vector3dVec pointCloud;
    cam2.reconstructPointCloud(imagePoints, pointCloud);
    T.transform(pointCloud, pointCloud); // Transformar si es necesario

    // Proyección inversa al modelo distorsionado
    if (params.size() == 6) {
        EnhancedCamera cam(params.data()); // EUCM
        cam.projectPointCloud(pointCloud, imagePoints);
    } else if (params.size() == 5) {
        UnifiedCamera cam(params.data()); // UCM
        cam.projectPointCloud(pointCloud, imagePoints);
    } else if (params.size() == 10) {
        MeiCamera cam(params.data()); // MEI
        cam.projectPointCloud(pointCloud, imagePoints);
    } else {
        cerr << "Parámetros de cámara no soportados." << endl;
        exit(1);
    }

    // Llenar mapas con verificación de validez
    auto it = imagePoints.begin();
    int orig_width  = static_cast<int>(pinhole_params[0]);
    int orig_height = static_cast<int>(pinhole_params[1]);

    for (int i = 0; i < mapX.rows; ++i) {
        for (int j = 0; j < mapX.cols; ++j, ++it) {
            float x = (*it)[0];
            float y = (*it)[1];
            if (x >= 0 && x < orig_width && y >= 0 && y < orig_height) {
                mapX(i, j) = x;
                mapY(i, j) = y;
            } else {
                mapX(i, j) = -1; // Marca como inválido
                mapY(i, j) = -1;
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Uso: " << argv[0] << " <archivo_json>" << endl;
        return 1;
    }

    // Leer archivo de configuración
    ptree root;
    read_json(argv[1], root);
    vector<double> paramsCam     = readVector<double>(root.get_child("camera_params"));
    vector<double> pinholeParams = readVector<double>(root.get_child("pinhole_params"));
    Transf T = readTransform(root.get_child("xi_eucm_pinhole"));

    // Generar mapas de remapeo
    Mat32f mapX, mapY;
    initRemap(paramsCam, pinholeParams, mapX, mapY, T);

    int count = 0;
    for (auto & x : root.get_child("image_names")) {
        Mat32f img = imread(x.second.get_value<string>(), 0); // gris
        Mat32f img2;

        // Remap con cv::INTER_NEAREST y sin estirar zonas inválidas
        remap(img, img2, mapX, mapY, cv::INTER_NEAREST);

        imwrite("img_" + to_string(count) + ".png", img2);

        count++;
    }

    return 0;
}
