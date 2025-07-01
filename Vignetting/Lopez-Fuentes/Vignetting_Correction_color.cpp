#include <iostream>
#include <cmath>
#include <string>
#include <windows.h> // Para CreateDirectory
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

// Prototipos de funciones
float calH(float a, float b, float c, Mat channel, float u0, float v0);
bool check(float a, float b, float c);
void optimizeChannel(Mat& channel, float& a_min, float& b_min, float& c_min, float u0, float v0);
Mat generarVinetaEstimada(Size tamano, vector<float> a_mins, vector<float> b_mins, vector<float> c_mins);

// Función para crear directorios en Windows
bool createDirectory(const string& path) {
    return CreateDirectoryA(path.c_str(), NULL) || GetLastError() == ERROR_ALREADY_EXISTS;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Uso: " << argv[0] << " <ruta_imagen> <u0> <v0> [mostrar_preview (0/1)]" << endl;
        return -1;
    }

    // Obtener u0 y v0 desde argumentos
    float u0, v0;
    try {
        u0 = stof(argv[2]);
        v0 = stof(argv[3]);
    } catch (const exception& e) {
        cerr << "Error: u0 y v0 deben ser valores numericos validos." << endl;
        return -1;
    }

    // Configurar paths de entrada
    string inputPath = argv[1];
    // Verificar si el archivo existe
    if (GetFileAttributesA(inputPath.c_str()) == INVALID_FILE_ATTRIBUTES) {
        cerr << "Error: La imagen de entrada no existe: " << inputPath << endl;
        return -1;
    }

    // Extraer nombre del archivo, extension y subcarpeta
    size_t lastSlash = inputPath.find_last_of("\\/");
    string filename = lastSlash == string::npos ? inputPath : inputPath.substr(lastSlash + 1);
    size_t dotPos = filename.find_last_of(".");
    string baseName = dotPos == string::npos ? filename : filename.substr(0, dotPos);
    string extension = dotPos == string::npos ? "" : filename.substr(dotPos);
    // Extraer nombre de la subcarpeta (dataset_name)
    size_t parentSlash = inputPath.substr(0, lastSlash).find_last_of("\\/");
    string datasetName = parentSlash == string::npos ? "" : inputPath.substr(parentSlash + 1, lastSlash - parentSlash - 1);

    // Construir ruta de salida
    string outputBaseDir = ".\\corregidas_lopez\\" + datasetName;
    if (!createDirectory(".\\corregidas_lopez")) {
        cerr << "Error al crear la carpeta: .\\corregidas_lopez" << endl;
        return -1;
    }
    if (!createDirectory(outputBaseDir)) {
        cerr << "Error al crear la carpeta: " << outputBaseDir << endl;
        return -1;
    }

    string correctedPath = outputBaseDir + "\\" + baseName + "_corrected" + extension;
    string vinetaPath = outputBaseDir + "\\" + baseName + "_vineta.png";

    // Cargar imagen en color
    Mat img = imread(inputPath, IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Error cargando la imagen: " << inputPath << endl;
        return -1;
    }

    // Separar canales RGB
    vector<Mat> canales;
    split(img, canales);

    // Optimizar parámetros para cada canal
    vector<float> a_mins(3), b_mins(3), c_mins(3);
    for (int i = 0; i < 3; ++i) {
        optimizeChannel(canales[i], a_mins[i], b_mins[i], c_mins[i], u0, v0);
        cout << "Canal " << i 
             << ": a=" << a_mins[i] 
             << " b=" << b_mins[i] 
             << " c=" << c_mins[i] << endl;
    }

    // Aplicar corrección a cada canal
    const int filas = img.rows;
    const int columnas = img.cols;
    const float diagonal = sqrt(u0 * u0 + v0 * v0);

    for (int c = 0; c < 3; ++c) {
        Mat& canal = canales[c];
        const float a = a_mins[c];
        const float b = b_mins[c];
        const float cc = c_mins[c];

        for (int fila = 0; fila < filas; ++fila) {
            uchar* pixel = canal.ptr<uchar>(fila);
            for (int col = 0; col < columnas; ++col) {
                const float dx = col - u0;
                const float dy = fila - v0;
                const float r = sqrt(dx*dx + dy*dy) / diagonal;
                const float r2 = r * r;
                const float factor = 1.0f + a*r2 + b*r2*r2 + cc*r2*r2*r2;
                
                const float valor = pixel[col] * factor;
                pixel[col] = saturate_cast<uchar>(valor);
            }
        }
    }

    // Combinar canales y guardar resultados
    Mat resultado;
    merge(canales, resultado);
    if (!imwrite(correctedPath, resultado)) {
        cerr << "Error al guardar la imagen corregida: " << correctedPath << endl;
        return -1;
    }

    // Generar y guardar viñeta estimada
    Mat vineta = generarVinetaEstimada(img.size(), a_mins, b_mins, c_mins);
    if (!imwrite(vinetaPath, vineta)) {
        cerr << "Error al guardar la viñeta estimada: " << vinetaPath << endl;
        return -1;
    }

    // Mostrar preview si se solicita
    if (argc >= 5 && atoi(argv[4]) != 0) {
        imshow("Imagen Original", img);
        imshow("Resultado Corregido", resultado);
        imshow("Vineta Estimada", vineta);
        waitKey(0);
    }

    cout << "Correccion completada:\n"
         << "Imagen corregida: " << correctedPath << "\n"
         << "Vineta estimada: " << vinetaPath << endl;

    return 0;
}

// Función para generar la viñeta estimada
Mat generarVinetaEstimada(Size tamano, vector<float> a_mins, vector<float> b_mins, vector<float> c_mins) {
    Mat vineta(tamano, CV_8UC3);
    const float u0 = tamano.width / 2.0f;
    const float v0 = tamano.height / 2.0f;
    const float diagonal = sqrt(u0 * u0 + v0 * v0);

    for (int fila = 0; fila < tamano.height; ++fila) {
        Vec3b* filaDatos = vineta.ptr<Vec3b>(fila);
        for (int col = 0; col < tamano.width; ++col) {
            const float dx = col - u0;
            const float dy = fila - v0;
            const float r = sqrt(dx*dx + dy*dy) / diagonal;
            const float r2 = r*r;
            
            Vec3f factores;
            for (int c = 0; c < 3; ++c) {
                factores[c] = 1.0f / (1.0f + a_mins[c]*r2 + b_mins[c]*r2*r2 + c_mins[c]*r2*r2*r2);
            }

            filaDatos[col] = Vec3b(
                saturate_cast<uchar>(factores[2] * 255),
                saturate_cast<uchar>(factores[1] * 255),
                saturate_cast<uchar>(factores[0] * 255)
            );
        }
    }
    
    return vineta;
}

void optimizeChannel(Mat& channel, float& a_min, float& b_min, float& c_min, float u0, float v0) {
    float a = 0, b = 0, c = 0;
    a_min = b_min = c_min = 0;
    float delta = 8.0f;
    float Hmin = calH(a, b, c, channel, u0, v0);

    while (delta > 1/256.0f) {
        float candidatos[6][3] = {
            {a+delta, b, c}, {a-delta, b, c},
            {a, b+delta, c}, {a, b-delta, c},
            {a, b, c+delta}, {a, b, c-delta}
        };

        for (int i = 0; i < 6; ++i) {
            const float a_temp = candidatos[i][0];
            const float b_temp = candidatos[i][1];
            const float c_temp = candidatos[i][2];
            
            if (check(a_temp, b_temp, c_temp)) {
                const float H = calH(a_temp, b_temp, c_temp, channel, u0, v0);
                if (H < Hmin) {
                    Hmin = H;
                    a_min = a_temp;
                    b_min = b_temp;
                    c_min = c_temp;
                }
            }
        }

        a = a_min;
        b = b_min;
        c = c_min;
        delta *= 0.5f;
    }
}

float calH(float a, float b, float c, Mat channel, float u0, float v0) {
    Mat floatChannel;
    channel.convertTo(floatChannel, CV_32F);
    
    const int filas = channel.rows;
    const int columnas = channel.cols;
    const float diagonal = sqrt(u0 * u0 + v0 * v0);

    for (int fila = 0; fila < filas; ++fila) {
        float* filaDatos = floatChannel.ptr<float>(fila);
        for (int col = 0; col < columnas; ++col) {
            const float dx = col - u0;
            const float dy = fila - v0;
            const float r = sqrt(dx*dx + dy*dy) / diagonal;
            const float r2 = r*r;
            const float r4 = r2*r2;
            const float r6 = r2*r2*r2;
            const float g = 1 + a*r2 + b*r4 + c*r6;
            filaDatos[col] *= g;
        }
    }

    Mat histograma = Mat::zeros(256, 1, CV_32F);
    for (int fila = 0; fila < filas; ++fila) {
        const float* filaDatos = floatChannel.ptr<float>(fila);
        for (int col = 0; col < columnas; ++col) {
            const float valor = 255.0f * log(1.0f + filaDatos[col]) / 8.0f;
            const int inferior = floor(valor);
            const int superior = ceil(valor);
            
            if (inferior >= 0 && inferior < 256)
                histograma.at<float>(inferior) += (1.0f - (valor - inferior));
            
            if (superior >= 0 && superior < 256)
                histograma.at<float>(superior) += (valor - inferior);
        }
    }

    Mat histSuavizado;
    const float kernel[] = {1, 2, 3, 4, 5, 4, 3, 2, 1};
    filter2D(histograma, histSuavizado, -1, Mat(1, 9, CV_32F, (void*)kernel));
    histSuavizado /= 25.0f;

    const float total = sum(histSuavizado)[0];
    float H = 0.0f;
    for (int i = 0; i < 256; ++i) {
        const float p = histSuavizado.at<float>(i) / total;
        if (p > 0) H += p * log(p);
    }
    
    return -H;
}

bool check(float a, float b, float c) {
    if (c == 0) {
        if (b == 0) return a > 0;
        if (b > 0) return a >= 0;
        return (-a <= 2*b) && (b < 0);
    }
    
    const float discriminante = 4*b*b - 12*a*c;
    if (discriminante <= 0) return c > 0;
    
    const float q_p = (-2*b + sqrt(discriminante)) / (6*c);
    const float q_m = (-2*b - sqrt(discriminante)) / (6*c);
    
    if (c > 0) return (q_p <= 0 || q_m >= 1);
    return (q_p >= 1 && q_m <= 0);
}