#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Nagłówki OpenCV
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>    
#include <opencv2/highgui.hpp>    
#include <opencv2/imgproc.hpp>    
#include <opencv2/core/utility.hpp> 

// Nagłówki systemowe
#include <stdio.h>
#include <iostream>
#include <string> 
#include <vector> // Potrzebne do listy plików

// Nagłówek MPI
#include <mpi.h>

// OBA KERNELE CUDA (diffAndThresholdKernel i erosionKernel)  bez zmian

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

__global__ void erosionKernel(unsigned char* dstMask,
    const unsigned char* srcMask,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
    {
        unsigned char minVal = 255;
        for (int ky = -1; ky <= 1; ++ky)
        {
            for (int kx = -1; kx <= 1; ++kx)
            {
                int neighborIdx = (y + ky) * width + (x + kx);
                unsigned char val = srcMask[neighborIdx];
                if (val < minVal) {
                    minVal = val;
                }
            }
        }
        int centerIdx = y * width + x;
        dstMask[centerIdx] = minVal;
    }
}


void checkCudaError(cudaError_t status, const char* msg)
{
    if (status != cudaSuccess) {
        fprintf(stderr, "Blad CUDA: %s: %s\n", msg, cudaGetErrorString(status));
        cudaDeviceReset();
        MPI_Abort(MPI_COMM_WORLD, status);
    }
}


int main(int argc, char* argv[])
{
    //INICJALIZACJA MPI
    int world_rank;
    int world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    
    std::vector<std::string> videoFiles = {
        "C:\\Users\\mnosel\\Downloads\\test0.mp4",
        "C:\\Users\\mnosel\\Downloads\\test1.mp4",
        "C:\\Users\\mnosel\\Downloads\\test2.mp4",
        "C:\\Users\\mnosel\\Downloads\\test3.mp4",
    };

    if (videoFiles.empty()) {
        if (world_rank == 0) { std::cerr << "BLAD: Lista plikow wideo jest pusta!" << std::endl; }
        MPI_Finalize();
        return -1;
    }

    //ROZDZIAŁ PRACY (Logika MPI)
    std::string myVideoFile = videoFiles[world_rank % videoFiles.size()];

    // wyświetlanie dla siatki 4x2
    const int DISPLAY_WIDTH = 480;   
    int displayHeight = 270;        

    std::string windowTitle_Orig = "Oryginal - Proces " + std::to_string(world_rank);
    std::string windowTitle_Mask = "Maska - Proces " + std::to_string(world_rank);

   
    cv::namedWindow(windowTitle_Orig, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(windowTitle_Mask, cv::WINDOW_AUTOSIZE);
    // ---------------------------------------------------

    std::cout << "[Proces " << world_rank << "/" << world_size << "] Rozpoczynam przetwarzanie: " << myVideoFile << std::endl;

    const int MOTION_THRESHOLD = 25;
    cv::VideoCapture cap(myVideoFile);
    if (!cap.isOpened()) {
        std::cerr << "[Proces " << world_rank << "] BLAD: Nie mozna otworzyc pliku: " << myVideoFile << std::endl;
        MPI_Finalize();
        return -1;
    }

    cv::Mat frame, grayFrame, motionMaskGPU;
    unsigned char* d_current = nullptr, * d_prev = nullptr;
    unsigned char* d_mask_raw = nullptr, * d_mask_eroded = nullptr;
    int width, height;
    size_t dataSize = 0;
    bool isFirstFrame = true;
    cv::TickMeter tm;
    int frameCounter = 0;

    while (true)
    {
        tm.start();
        cap.read(frame);
        if (frame.empty()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            isFirstFrame = true;
            continue;
        }

        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        if (isFirstFrame)
        {
            width = grayFrame.cols;
            height = grayFrame.rows;
            dataSize = width * height * sizeof(unsigned char);
            checkCudaError(cudaMalloc((void**)&d_current, dataSize), "cudaMalloc d_current");
            checkCudaError(cudaMalloc((void**)&d_prev, dataSize), "cudaMalloc d_prev");
            checkCudaError(cudaMalloc((void**)&d_mask_raw, dataSize), "cudaMalloc d_mask_raw");
            checkCudaError(cudaMalloc((void**)&d_mask_eroded, dataSize), "cudaMalloc d_mask_eroded");
            checkCudaError(cudaMemset(d_mask_eroded, 0, dataSize), "cudaMemset d_mask_eroded");
            motionMaskGPU.create(height, width, CV_8UC1);
            checkCudaError(cudaMemcpy(d_prev, grayFrame.data, dataSize, cudaMemcpyHostToDevice), "cudaMemcpy d_prev (first frame)");

            displayHeight = (int)((double)height / width * DISPLAY_WIDTH);

           
            int grid_x = world_rank % 2; // 0 lub 1 (kolumna)
            int grid_y = world_rank / 2; // 0 lub 1 (wiersz)

            // Procesy 0,1,2,3... -> (0,0), (1,0), (0,1), (1,1)
            int posX_Orig = grid_x * (DISPLAY_WIDTH + 10); // +10 na ramkę
            int posX_Mask = posX_Orig + DISPLAY_WIDTH + 10;
            int posY_All = grid_y * (displayHeight + 40); // +40 na pasek tytułowy

            cv::moveWindow(windowTitle_Orig, posX_Orig, posY_All);
            cv::moveWindow(windowTitle_Mask, posX_Mask, posY_All);
            // -------------------------------------------------

            isFirstFrame = false;
            continue;
        }

        
        checkCudaError(cudaMemcpy(d_current, grayFrame.data, dataSize, cudaMemcpyHostToDevice), "cudaMemcpy d_current");
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
        diffAndThresholdKernel << <numBlocks, threadsPerBlock >> > (d_mask_raw, d_current, d_prev, width, height, MOTION_THRESHOLD);
        erosionKernel << <numBlocks, threadsPerBlock >> > (d_mask_eroded, d_mask_raw, width, height);
        checkCudaError(cudaGetLastError(), "Kernel launch failure");
        checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        checkCudaError(cudaMemcpy(motionMaskGPU.data, d_mask_eroded, dataSize, cudaMemcpyDeviceToHost), "cudaMemcpy d_mask (D2H)");
        checkCudaError(cudaMemcpy(d_prev, d_current, dataSize, cudaMemcpyDeviceToDevice), "cudaMemcpy d_prev (D2D)");

        tm.stop();
        double fps = tm.getFPS();
        std::string fpsText = "FPS: " + std::to_string((int)fps);
        cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        
        cv::Mat smallFrame, smallMask;
        cv::resize(frame, smallFrame, cv::Size(DISPLAY_WIDTH, displayHeight));
        cv::resize(motionMaskGPU, smallMask, cv::Size(DISPLAY_WIDTH, displayHeight), 0, 0, cv::INTER_NEAREST);

        cv::imshow(windowTitle_Orig, smallFrame);
        cv::imshow(windowTitle_Mask, smallMask);
        // -------------------------------------------------

        // === ZSYNCHRONIZOWANA OBSŁUGA KLAWISZY (MPI) ===
        int key = cv::waitKey(1);
        int local_save_command = (key == 's' || key == 'S') ? 1 : 0;
        int local_quit_command = (key == 27) ? 1 : 0;
        int global_save_command = 0;
        int global_quit_command = 0;

        MPI_Allreduce(&local_save_command, &global_save_command, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&local_quit_command, &global_quit_command, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (global_quit_command == 1) {
            break;
        }

        if (global_save_command == 1)
        {
            std::string savePath = "C:\\Users\\mnosel\\Desktop\\";
            std::string originalName = savePath + "proces_" + std::to_string(world_rank) + "_klatka_" + std::to_string(frameCounter) + "_oryginal.png";
            std::string maskName = savePath + "proces_" + std::to_string(world_rank) + "_klatka_" + std::to_string(frameCounter) + "_maska_ruch.png";
            cv::imwrite(originalName, frame); 
            cv::imwrite(maskName, motionMaskGPU); 
            if (world_rank == 0) { std::cout << "ZAPISANO klatki (wszystkie procesy)..." << std::endl; }
            frameCounter++;
        }
    }

    std::cout << "[Proces " << world_rank << "] Zamykanie..." << std::endl;
    cudaFree(d_current);
    cudaFree(d_prev);
    cudaFree(d_mask_raw);
    cudaFree(d_mask_eroded);
    cudaDeviceReset();

    cap.release();
    cv::destroyAllWindows();

    // === 4. FINALIZACJA MPI ===
    MPI_Finalize();
    return 0;
}