#include <iostream>
#include <opencv2/opencv.hpp>
#include "ColorCorrection.hpp"
#include <vector>
#include <string>

using namespace std;
using namespace cv;

int VignettingCorrect(cv::Mat& pImage, const string& vignettePath, double u0, double v0) 
{
    int ht = pImage.rows;
    int wd = pImage.cols;
    int nChannels = pImage.channels();

    if (nChannels != 3) {
        cerr << "Error: La imagen debe tener 3 canales (BGR)" << endl;
        return -1;
    }

    cout << "Centro optico: u0 = " << u0 << ", v0 = " << v0 << endl;

    double ratio = 1;
    if (wd > 75)
        ratio = 75.0 / double(wd);

    int sht = static_cast<int>(ht * ratio + 0.5);
    int swd = static_cast<int>(wd * ratio + 0.5);
    cv::Mat pSmallImage;
    cv::resize(pImage, pSmallImage, cv::Size(swd, sht));

    cv::Mat pGrayImage;
    cv::cvtColor(pSmallImage, pGrayImage, cv::COLOR_BGR2GRAY);

    unsigned char* pImageBuffer = (unsigned char*)malloc(sht * swd);
    if (!pImageBuffer) {
        cerr << "Error: No se pudo asignar memoria para pImageBuffer." << endl;
        return -1;
    }
    for (int j = 0; j < sht; j++) {
        for (int i = 0; i < swd; i++) {
            pImageBuffer[j * swd + i] = pGrayImage.at<uchar>(j, i);
        }
    }

    vector<double> vp;
    int flag = VignettingCorrectionUsingRG(pImageBuffer, sht, swd, vp);

    if (flag == 0) {
        int nV = static_cast<int>(vp.size());
        for (int i = 0; i < nV; i++) {
            vp[i] = exp(vp[i]);
        }

        double maxVr = vp[0];
        for (int i = 0; i < nV; i++) {
            vp[i] /= maxVr;
        }

        for (int j = 0; j < ht; j++) {
            for (int i = 0; i < wd; i++) {
                double u = i * ratio;
                double v = j * ratio;
                double cx = u - u0 * ratio;
                double cy = v - v0 * ratio;
                double radius = sqrt(cx * cx + cy * cy);

                double vValue = 1.0;
                int nR = static_cast<int>(radius);
                if (nR == 0) {
                    vValue = vp[0];
                } else if (nR < nV) {
                    double dr = radius - nR;
                    vValue = vp[nR - 1] * (1 - dr) + vp[nR] * dr;
                } else {
                    vValue = vp[nV - 1];
                }

                double scale = 1.0 / vValue;

                for (int c = 0; c < nChannels; c++) {
                    int pixelValue = static_cast<int>(pImage.at<cv::Vec3b>(j, i)[c] * scale);
                    pImage.at<cv::Vec3b>(j, i)[c] = std::min(255, pixelValue);
                }
            }
        }

        cv::Mat Estimate_temp(sht, swd, CV_8UC1);
        double scaled_u0 = u0 * ratio;
        double scaled_v0 = v0 * ratio;
        for (int i = 0; i < sht; i++) {
            uchar* data = Estimate_temp.ptr<uchar>(i);
            for (int j = 0; j < swd; j++) {
                double cx = j - scaled_u0;
                double cy = i - scaled_v0;
                int r = static_cast<int>(sqrt(cx * cx + cy * cy) + 0.5);
                if (r > 0 && r < nV + 1 && vp[r - 1] < 1)
                    data[j] = static_cast<uchar>(255 * vp[r - 1]);
                else
                    data[j] = 255;
            }
        }

        cv::Mat Estimate;
        cv::resize(Estimate_temp, Estimate, cv::Size(wd, ht));
        if (!cv::imwrite(vignettePath, Estimate)) {
            cerr << "Error: No se pudo guardar el mapa de viñeta: " << vignettePath << endl;
            free(pImageBuffer);
            return -1;
        }
    }

    free(pImageBuffer);
    return flag;
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        cout << "Uso: " << argv[0] << " <ruta_imagen> <u0> <v0> [mostrar_preview (0/1)]" << endl;
        cout << "Ejemplo: " << argv[0] << " .\\imagenes_vineteadas\\dataset1\\chessboard_01.jpg 820 616 1" << endl;
        return -1;
    }

    double u0, v0;
    try {
        u0 = std::stod(argv[2]);
        v0 = std::stod(argv[3]);
    } catch (const std::exception& e) {
        cerr << "Error: u0 y v0 deben ser valores numéricos." << endl;
        return -1;
    }

    bool showPreview = false;
    if (argc >= 5) {
        showPreview = (atoi(argv[4]) != 0);
    }

    string inputPath = argv[1];
    size_t lastSlash = inputPath.find_last_of("\\/");
    string filename = lastSlash == string::npos ? inputPath : inputPath.substr(lastSlash + 1);
    size_t dotPos = filename.find_last_of(".");
    string baseName = dotPos == string::npos ? filename : filename.substr(0, dotPos);
    string extension = dotPos == string::npos ? "" : filename.substr(dotPos);
    size_t parentSlash = inputPath.substr(0, lastSlash).find_last_of("\\/");
    string datasetName = parentSlash == string::npos ? "" : inputPath.substr(parentSlash + 1, lastSlash - parentSlash - 1);

    string outputDir = "corregidas_zheng/" + datasetName + "/";
    string correctedPath = outputDir + baseName + "_corrected" + extension;
    string vignettePath = outputDir + baseName + "_vineta.png";

    cv::Mat img = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Error: No se pudo cargar la imagen: " << inputPath << endl;
        return -1;
    }

    int result = VignettingCorrect(img, vignettePath, u0, v0);

    if (result == 0) {
        if (!cv::imwrite(correctedPath, img)) {
            cerr << "Error: No se pudo guardar la imagen corregida: " << correctedPath << endl;
            return -1;
        }
        
        if (showPreview) {
            cv::namedWindow("Imagen Original", cv::WINDOW_NORMAL);
            cv::namedWindow("Imagen Corregida", cv::WINDOW_NORMAL);
            cv::Mat original = cv::imread(inputPath, cv::IMREAD_COLOR);
            cv::imshow("Imagen Original", original);
            cv::imshow("Imagen Corregida", img);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
        
        cout << "Proceso completado exitosamente:\n"
             << "Imagen corregida: " << correctedPath << "\n"
             << "Mapa de viñeta: " << vignettePath << endl;
    } else {
        cerr << "Error durante la corrección de viñeta" << endl;
        return -1;
    }

    return 0;
}