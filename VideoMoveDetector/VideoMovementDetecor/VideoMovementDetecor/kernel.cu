#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Nagłówki OpenCV
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>    
#include <opencv2/highgui.hpp>    
#include <opencv2/imgproc.hpp>    
#include <opencv2/core/utility.hpp> // Dla cv::TickMeter

// Nagłówek OpenMP
#include <omp.h>

#include <stdio.h>
#include <iostream>
#include <string> // Dla std::to_string

// ======================================================================
// KERNEL CUDA (bez zmian)
// ======================================================================

__global__ void diffAndThresholdKernel(unsigned char* mask,
    const unsigned char* current,
    const unsigned char* prev,
    int width, int height,
    int threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        int diff = abs((int)current[idx] - (int)prev[idx]);
        mask[idx] = (diff > threshold) ? 255 : 0;
    }
}

// ======================================================================
// FUNKCJA OpenMP (bez zmian)
// ======================================================================

void manualErosionOpenMP(const cv::Mat& src, cv::Mat& dst)
{
    dst.create(src.size(), src.type());
    int rows = src.rows;
    int cols = src.cols;

#pragma omp parallel for collapse(2)
    for (int y = 1; y < rows - 1; ++y)
    {
        for (int x = 1; x < cols - 1; ++x)
        {
            unsigned char minVal = 255;
            for (int ky = -1; ky <= 1; ++ky)
            {
                for (int kx = -1; kx <= 1; ++kx)
                {
                    unsigned char val = src.at<unsigned char>(y + ky, x + kx);
                    if (val < minVal) {
                        minVal = val;
                    }
                }
            }
            dst.at<unsigned char>(y, x) = minVal;
        }
    }
}

// Funkcja pomocnicza do sprawdzania błędów CUDA
void checkCudaError(cudaError_t status, const char* msg)
{
    if (status != cudaSuccess) {
        fprintf(stderr, "Blad CUDA: %s: %s\n", msg, cudaGetErrorString(status));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

// ======================================================================
// GŁÓWNA FUNKCJA PROGRAMU (ZMIANY)
// ======================================================================

int main()
{
    const int MOTION_THRESHOLD = 25;

    // --- Ustawienie rozdzielczości (DO TESTÓW FPS) ---
    // Domyślne to zazwyczaj 640x480. Możesz spróbować zmienić na 1280x720.


    cv::VideoCapture cap("C:\\Users\\mnosel\\Downloads\\test.mp4");
    if (!cap.isOpened())
    {
        std::cerr << "BLAD: Nie mozna otworzyc kamery internetowej!" << std::endl;
        std::cin.get();
        return -1;
    }

    // Ustawienie żądanej rozdzielczości kamery
    

    std::cout << "Otwarto kamere. Rozpoczynanie przetwarzania..." << std::endl;
    std::cout << "Nacisnij 'ESC', aby zakonczyc." << std::endl;
    std::cout << "Nacisnij 's', aby zapisac klatki." << std::endl;

    cv::Mat frame;
    cv::Mat grayFrame;
    cv::Mat motionMaskGPU;
    cv::Mat motionMaskOMP;

    unsigned char* d_current = nullptr;
    unsigned char* d_prev = nullptr;
    unsigned char* d_mask = nullptr;

    int width, height;
    size_t dataSize = 0;
    bool isFirstFrame = true;

    // --- NOWOŚĆ: Pomiar FPS ---
    cv::TickMeter tm; // Obiekt do mierzenia czasu
    int frameCounter = 0; // Licznik klatek do zapisu

    while (true)
    {
        tm.start(); // <-- Rozpocznij pomiar czasu
        cap.read(frame);
        if (frame.empty()) {
            std::cout << "Koniec pliku wideo. Zapetlanie..." << std::endl;
            // Przewiń wideo z powrotem na klatkę 0
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            // Zresetuj 'isFirstFrame', aby poprawnie załadować bufor 'd_prev'
            isFirstFrame = true;
            continue; // Przejdź do następnej iteracji (wczyta nową klatkę 0)
        }

        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        if (isFirstFrame)
        {
            width = grayFrame.cols;
            height = grayFrame.rows;
            // Sprawdź, czy kamera faktycznie ustawiła żądaną rozdzielczość
            std::cout << "Rozdzielczosc przetwarzania: " << width << "x" << height << std::endl;

            dataSize = width * height * sizeof(unsigned char);

            checkCudaError(cudaMalloc((void**)&d_current, dataSize), "cudaMalloc d_current");
            checkCudaError(cudaMalloc((void**)&d_prev, dataSize), "cudaMalloc d_prev");
            checkCudaError(cudaMalloc((void**)&d_mask, dataSize), "cudaMalloc d_mask");

            motionMaskGPU.create(height, width, CV_8UC1);
            motionMaskOMP.create(height, width, CV_8UC1);

            checkCudaError(cudaMemcpy(d_prev, grayFrame.data, dataSize, cudaMemcpyHostToDevice), "cudaMemcpy d_prev (first frame)");
            isFirstFrame = false;
            continue;
        }

        // === POCZĄTEK PRZETWARZANIA ===
        checkCudaError(cudaMemcpy(d_current, grayFrame.data, dataSize, cudaMemcpyHostToDevice), "cudaMemcpy d_current");

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        diffAndThresholdKernel << <numBlocks, threadsPerBlock >> > (d_mask, d_current, d_prev, width, height, MOTION_THRESHOLD);

        checkCudaError(cudaGetLastError(), "diffAndThresholdKernel launch");
        checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        checkCudaError(cudaMemcpy(motionMaskGPU.data, d_mask, dataSize, cudaMemcpyDeviceToHost), "cudaMemcpy d_mask (D2H)");

        manualErosionOpenMP(motionMaskGPU, motionMaskOMP);

        checkCudaError(cudaMemcpy(d_prev, d_current, dataSize, cudaMemcpyDeviceToDevice), "cudaMemcpy d_prev (D2D)");
        // === KONIEC PRZETWARZANIA ===

        tm.stop(); // <-- Zakończ pomiar czasu

        // --- NOWOŚĆ: Wyświetlanie FPS ---
        // Obliczamy FPS (używamy średniej kroczącej dla stabilności)
        double fps = tm.getFPS();
        std::string fpsText = "FPS: " + std::to_string((int)fps);

        // Rysuj tekst FPS na oryginalnej klatce
        cv::putText(frame,
            fpsText,
            cv::Point(10, 30), // Pozycja (X, Y)
            cv::FONT_HERSHEY_SIMPLEX,
            1.0, // Skala czcionki
            cv::Scalar(0, 255, 0), // Kolor (zielony)
            2); // Grubość

        // Wyświetlanie wyników
        cv::imshow("Oryginal (Kamera) z FPS", frame);
        cv::imshow("Maska Ruchu (z GPU)", motionMaskGPU);
        cv::imshow("Maska Ruchu po Morfologii (OpenMP)", motionMaskOMP);

        // Obsługa klawiszy
        int key = cv::waitKey(1);
        if (key == 27) { // ESC
            break;
        }
        // --- NOWOŚĆ: Zapisywanie klatek ---
        else if (key == 's' || key == 'S')
        {
            std::string originalName = "klatka_" + std::to_string(frameCounter) + "_oryginal.png";
            std::string maskName = "klatka_" + std::to_string(frameCounter) + "_maska_ruch.png";

            cv::imwrite(originalName, frame);
            cv::imwrite(maskName, motionMaskOMP); // Zapisujemy końcową maskę po morfologii

            std::cout << "ZAPISANO: " << originalName << " oraz " << maskName << std::endl;
            frameCounter++;
        }
    }

    // --- Sprzątanie ---
    std::cout << "Zamykanie..." << std::endl;
    cudaFree(d_current);
    cudaFree(d_prev);
    cudaFree(d_mask);
    cudaDeviceReset();

    cap.release();
    cv::destroyAllWindows();

    return 0;
}