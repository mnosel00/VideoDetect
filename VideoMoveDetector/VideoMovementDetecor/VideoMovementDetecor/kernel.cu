#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>    
#include <opencv2/highgui.hpp>    
#include <opencv2/imgproc.hpp>    
#include <opencv2/core/utility.hpp> 
#include <omp.h> 
#include <stdio.h>
#include <iostream>
#include <string> 
#include <vector>
#include <iomanip> // Do ładnej tabelki

// --- KERNEL CUDA (BEZ ZMIAN) ---
__global__ void diffAndThresholdKernel(unsigned char* mask, const unsigned char* current, const unsigned char* prev, int width, int height, int threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        int diff = abs((int)current[idx] - (int)prev[idx]);
        mask[idx] = (diff > threshold) ? 255 : 0;
    }
}

// --- FUNKCJA OPENMP (BEZ ZMIAN) ---
void manualErosionOpenMP(const cv::Mat& src, cv::Mat& dst)
{
    dst.create(src.size(), src.type());
    int rows = src.rows;
    int cols = src.cols;
#pragma omp parallel for collapse(2)
    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            unsigned char minVal = 255;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    unsigned char val = src.at<unsigned char>(y + ky, x + kx);
                    if (val < minVal) minVal = val;
                }
            }
            dst.at<unsigned char>(y, x) = minVal;
        }
    }
}

void checkCudaError(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Blad CUDA: %s: %s\n", msg, cudaGetErrorString(status));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

int main()
{
    // ZMIEŃ ŚCIEŻKĘ DO SWOJEGO PLIKU!
    std::string videoPath = "C:\\Users\\mnosel\\Downloads\\test0.mp4";
    const int MOTION_THRESHOLD = 25;

    // Lista testów: ile wątków chcemy sprawdzić
    std::vector<int> threadCounts = { 1, 2, 4, 8, 12 };
    const int FRAMES_TO_TEST = 150; // Ile klatek testować dla każdego przypadku

    std::cout << "=================================================" << std::endl;
    std::cout << "  AUTOMATYCZNY TEST WYDAJNOSCI (OpenMP vs Threads)" << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "Ladowanie wideo..." << std::endl;

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) { std::cerr << "BLAD: Nie mozna otworzyc wideo!" << std::endl; return -1; }

    cv::Mat frame, grayFrame, motionMaskGPU, motionMaskOMP;
    cap.read(frame);
    int width = frame.cols;
    int height = frame.rows;
    size_t dataSize = width * height * sizeof(unsigned char);

    unsigned char* d_current, * d_prev, * d_mask;
    checkCudaError(cudaMalloc((void**)&d_current, dataSize), "Alloc current");
    checkCudaError(cudaMalloc((void**)&d_prev, dataSize), "Alloc prev");
    checkCudaError(cudaMalloc((void**)&d_mask, dataSize), "Alloc mask");

    motionMaskGPU.create(height, width, CV_8UC1);
    motionMaskOMP.create(height, width, CV_8UC1);

    // Przygotowanie pierwszej klatki
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    checkCudaError(cudaMemcpy(d_prev, grayFrame.data, dataSize, cudaMemcpyHostToDevice), "Copy prev");

    // Tabela na wyniki
    std::vector<double> results;

    std::cout << "Start testow. Okna wideo sa UKRYTE dla lepszej dokladnosci." << std::endl;
    std::cout << "Prosze czekac..." << std::endl;

    for (int threads : threadCounts)
    {
        // Ustawienie liczby wątków
        omp_set_num_threads(threads);

        std::cout << "-> Testowanie: " << threads << " watkow... ";

        // Reset wideo
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);

        cv::TickMeter tm;
        tm.start();

        for (int i = 0; i < FRAMES_TO_TEST; i++)
        {
            cap.read(frame);
            if (frame.empty()) { cap.set(cv::CAP_PROP_POS_FRAMES, 0); cap.read(frame); }

            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
            checkCudaError(cudaMemcpy(d_current, grayFrame.data, dataSize, cudaMemcpyHostToDevice), "H2D");

            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((width + 15) / 16, (height + 15) / 16);

            diffAndThresholdKernel << <numBlocks, threadsPerBlock >> > (d_mask, d_current, d_prev, width, height, MOTION_THRESHOLD);
            cudaDeviceSynchronize();

            checkCudaError(cudaMemcpy(motionMaskGPU.data, d_mask, dataSize, cudaMemcpyDeviceToHost), "D2H");

            // TO JEST MIERZONE (OpenMP CPU)
            manualErosionOpenMP(motionMaskGPU, motionMaskOMP);

            checkCudaError(cudaMemcpy(d_prev, d_current, dataSize, cudaMemcpyDeviceToDevice), "D2D");
        }

        tm.stop();
        double fps = FRAMES_TO_TEST / tm.getTimeSec();
        results.push_back(fps);
        std::cout << "Wynik: " << (int)fps << " FPS" << std::endl;
    }

    // --- WYPISANIE WYNIKÓW ---
    std::cout << "\n\n========================================" << std::endl;
    std::cout << "       WYNIKI DO SPRAWOZDANIA           " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::left << std::setw(15) << "Liczba Watkow" << " | " << "FPS" << std::endl;
    std::cout << "----------------|-------" << std::endl;

    for (size_t i = 0; i < threadCounts.size(); i++) {
        std::cout << std::left << std::setw(15) << threadCounts[i] << " | " << results[i] << std::endl;
    }
    std::cout << "========================================" << std::endl;

    // Sprzątanie
    cudaFree(d_current); cudaFree(d_prev); cudaFree(d_mask);
    std::cout << "\nNacisnij Enter aby zakonczyc...";
    std::cin.get();
    return 0;
}