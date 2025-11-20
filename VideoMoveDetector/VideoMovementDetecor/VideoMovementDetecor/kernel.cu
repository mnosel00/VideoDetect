#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>    
#include <opencv2/highgui.hpp>    
#include <opencv2/imgproc.hpp>    
#include <opencv2/core/utility.hpp> 
#include <stdio.h>
#include <iostream>
#include <string> 

// --- KERNELE CUDA (BEZ ZMIAN) ---
__global__ void diffAndThresholdKernel(unsigned char* mask, const unsigned char* current, const unsigned char* prev, int width, int height, int threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        int diff = abs((int)current[idx] - (int)prev[idx]);
        mask[idx] = (diff > threshold) ? 255 : 0;
    }
}

__global__ void erosionKernel(unsigned char* dstMask, const unsigned char* srcMask, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        unsigned char minVal = 255;
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                int neighborIdx = (y + ky) * width + (x + kx);
                unsigned char val = srcMask[neighborIdx];
                if (val < minVal) minVal = val;
            }
        }
        dstMask[y * width + x] = minVal;
    }
}

void checkCudaError(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Blad CUDA: %s: %s\n", msg, cudaGetErrorString(status));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int MOTION_THRESHOLD = 25;
    cv::VideoCapture cap("C:\\Users\\mnosel\\Downloads\\test0.mp4");
    if (!cap.isOpened()) { std::cerr << "BLAD WIDEO" << std::endl; return -1; }

    std::cout << "START: Full GPU Mode" << std::endl;
    std::cout << "Dane o FPS beda wypisywane co 100 klatek." << std::endl;

    cv::Mat frame, grayFrame, motionMaskGPU;
    unsigned char* d_current, * d_prev, * d_mask_raw, * d_mask_eroded;

    cap.read(frame);
    int width = frame.cols;
    int height = frame.rows;
    size_t dataSize = width * height;

    checkCudaError(cudaMalloc((void**)&d_current, dataSize), "Alloc");
    checkCudaError(cudaMalloc((void**)&d_prev, dataSize), "Alloc");
    checkCudaError(cudaMalloc((void**)&d_mask_raw, dataSize), "Alloc");
    checkCudaError(cudaMalloc((void**)&d_mask_eroded, dataSize), "Alloc");
    checkCudaError(cudaMemset(d_mask_eroded, 0, dataSize), "Zero");

    motionMaskGPU.create(height, width, CV_8UC1);
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    checkCudaError(cudaMemcpy(d_prev, grayFrame.data, dataSize, cudaMemcpyHostToDevice), "Copy 1st");

    cv::TickMeter tm;
    tm.start(); // Start licznika globalnego
    int frameCounter = 0;
    int framesForAverage = 0; // Licznik do uśredniania

    while (true) {
        cap.read(frame);
        if (frame.empty()) { cap.set(cv::CAP_PROP_POS_FRAMES, 0); continue; }

        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        checkCudaError(cudaMemcpy(d_current, grayFrame.data, dataSize, cudaMemcpyHostToDevice), "H2D");

        dim3 threads(16, 16);
        dim3 blocks((width + 15) / 16, (height + 15) / 16);

        diffAndThresholdKernel << <blocks, threads >> > (d_mask_raw, d_current, d_prev, width, height, MOTION_THRESHOLD);
        erosionKernel << <blocks, threads >> > (d_mask_eroded, d_mask_raw, width, height);
        cudaDeviceSynchronize();

        checkCudaError(cudaMemcpy(motionMaskGPU.data, d_mask_eroded, dataSize, cudaMemcpyDeviceToHost), "D2H");
        checkCudaError(cudaMemcpy(d_prev, d_current, dataSize, cudaMemcpyDeviceToDevice), "D2D");

        // --- WYPISYWANIE DANYCH DO WYKRESU ---
        framesForAverage++;
        if (framesForAverage == 100) {
            tm.stop();
            double avgFps = 100.0 / tm.getTimeSec();
            std::cout << "[DANE DO WYKRESU] Sredni FPS (GPU): " << avgFps << std::endl;
            tm.reset();
            tm.start();
            framesForAverage = 0;
        }

        // Wyświetlanie (możesz zakomentować, jeśli chcesz super dokładny wynik bez GUI)
        cv::imshow("Full GPU", motionMaskGPU);

        int key = cv::waitKey(1);
        if (key == 27) break;

        // --- POPRAWIONY ZAPIS 3 ZDJEĆ ---
        else if (key == 's' || key == 'S') {
            std::string path = "C:\\Users\\mnosel\\Desktop\\";
            cv::Mat h_mask_raw(height, width, CV_8UC1);
            cudaMemcpy(h_mask_raw.data, d_mask_raw, dataSize, cudaMemcpyDeviceToHost); // Pobierz brudną maskę

            cv::imwrite(path + "gpu_1_oryginal.png", frame);
            cv::imwrite(path + "gpu_2_szum.png", h_mask_raw);
            cv::imwrite(path + "gpu_3_czyste.png", motionMaskGPU);
            std::cout << "Zapisano 3 zdjecia!" << std::endl;
        }
    }

    cudaFree(d_current); cudaFree(d_prev); cudaFree(d_mask_raw); cudaFree(d_mask_eroded);
    return 0;
}