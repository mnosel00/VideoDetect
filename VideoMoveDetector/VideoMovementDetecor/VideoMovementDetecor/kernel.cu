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


// KERNEL CUDA (bez zmian)

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


// FUNKCJA OpenMP 
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

//sprawdzanie błędów CUDA
void checkCudaError(cudaError_t status, const char* msg)
{
    if (status != cudaSuccess) {
        fprintf(stderr, "Blad CUDA: %s: %s\n", msg, cudaGetErrorString(status));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

int main()
{
    const int MOTION_THRESHOLD = 25;
    
    cv::VideoCapture cap("C:\\Users\\mnosel\\Downloads\\test0.mp4");
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

    //Pomiar FPS ---
    cv::TickMeter tm; // Obiekt do mierzenia czasu
    int frameCounter = 0; // Licznik klatek do zapisu

    while (true)
    {
        tm.start(); //pomiar czasu
        cap.read(frame);

        if (frame.empty()) {
            std::cout << "Reset filmu" << std::endl;
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            isFirstFrame = true;
            continue; //wczytanie nową klatkę 0)
        } //reset filmu

        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        if (isFirstFrame)
        {
            width = grayFrame.cols;
            height = grayFrame.rows;

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

        tm.stop(); //Koniec pomiary czasu

        //Wyświetlanie FPS
        // FPS (używamy średniej kroczącej dla stabilności)
        double fps = tm.getFPS();
        std::string fpsText = "FPS: " + std::to_string((int)fps);

        // FPS na  klatce
        cv::putText(frame,
            fpsText,
            cv::Point(10, 30), // Pozycja (X, Y)
            cv::FONT_HERSHEY_SIMPLEX,
            1.0, // Skala czcionki
            cv::Scalar(0, 255, 0), // Kolor (zielony)
            2); // Grubość

       
        cv::imshow("Oryginal (Kamera) z FPS", frame);
        cv::imshow("Maska Ruchu (z GPU)", motionMaskGPU);
        cv::imshow("Maska Ruchu po Morfologii (OpenMP)", motionMaskOMP);

       
        int key = cv::waitKey(1);
        if (key == 27) { // ESC
            break;
        }
       
        //ZAPISYWANIe (3 KLATKI)
        else if (key == 's' || key == 'S')
        {
            
            std::string savePath = "C:\\Users\\mnosel\\Desktop\\";

            
            std::string originalName = savePath + "klatka_" + std::to_string(frameCounter) + "_1_oryginal.png";
            std::string maskRawName = savePath + "klatka_" + std::to_string(frameCounter) + "_2_maska_z_szumem.png";
            std::string maskCleanName = savePath + "klatka_" + std::to_string(frameCounter) + "_3_maska_oczyszczona.png";

            // Czy obrazy nie są puste
            if (frame.empty() || motionMaskGPU.empty() || motionMaskOMP.empty()) {
                std::cerr << "BLAD: Proba zapisu pustej klatki!" << std::endl;
            }
            else
            {
                // Save wszystkie trzy obrazy
                bool success1 = cv::imwrite(originalName, frame);        
                bool success2 = cv::imwrite(maskRawName, motionMaskGPU); 
                bool success3 = cv::imwrite(maskCleanName, motionMaskOMP); 

                if (success1 && success2 && success3) {
                    std::cout << "ZAPISANO POMYSLNIE (3 pliki) do: " << savePath << std::endl;
                }
                else {
                    std::cerr << "BLAD ZAPISU: Nie udalo sie zapisac wszystkich klatek. Sprawdz sciezke: " << savePath << std::endl;
                }

                frameCounter++;
            }
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