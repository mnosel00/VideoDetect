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

// ======================================================================
// OBA KERNELE CUDA (diffAndThresholdKernel i erosionKernel)
// POZOSTAJĄ BEZ ZMIAN - (nie wklejam ich tu ponownie dla zwięzłości,
// ale upewnij się, że są w Twoim pliku - po prostu je zostaw)
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


// Funkcja pomocnicza do sprawdzania błędów CUDA
void checkCudaError(cudaError_t status, const char* msg)
{
    if (status != cudaSuccess) {
        fprintf(stderr, "Blad CUDA: %s: %s\n", msg, cudaGetErrorString(status));
        cudaDeviceReset();
        // W MPI lepiej nie robić exit(), tylko zakończyć program
        MPI_Abort(MPI_COMM_WORLD, status);
    }
}

// ======================================================================
// GŁÓWNA FUNKCJA PROGRAMU (ZMIANY MPI)
// ======================================================================

// ZMIANA: main musi teraz przyjmować argumenty dla MPI
int main(int argc, char* argv[])
{
    // === 1. INICJALIZACJA MPI ===
    int world_rank; // ID tego procesu (np. 0, 1, 2...)
    int world_size; // Całkowita liczba procesów (ile kopii uruchomiliśmy)

    MPI_Init(&argc, &argv); // Inicjujemy MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Pobieramy nasze ID
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Pobieramy łączną liczbę procesów

    // === 2. LISTA ZADAŃ (PLIKI WIDEO) ===
    // Każdy proces musi znać całą listę.
    // Upewnij się, że masz te pliki i ścieżki są poprawne!
    std::vector<std::string> videoFiles = {
        "C:\\Users\\mnosel\\Downloads\\test0.mp4",
        "C:\\Users\\mnosel\\Downloads\\test1.mp4",
        "C:\\Users\\mnosel\\Downloads\\test2.mp4",
        "C:\\Users\\mnosel\\Downloads\\test3.mp4",
        // Możesz dodać więcej filmów, jeśli chcesz
    };

    if (videoFiles.empty()) {
        if (world_rank == 0) { // Tylko proces 0 (główny) wypisze błąd
            std::cerr << "BLAD: Lista plikow wideo jest pusta!" << std::endl;
        }
        MPI_Finalize();
        return -1;
    }

    // === 3. ROZDZIAŁ PRACY (Logika MPI) ===
    // Używamy naszego 'rank' (ID), aby wybrać plik wideo.
    // Operator modulo (%) zapewnia, że zadania zostaną rozdzielone,
    // nawet jeśli mamy więcej procesów niż filmów.
    std::string myVideoFile = videoFiles[world_rank % videoFiles.size()];

    // Tworzymy unikalny tytuł okna dla każdego procesu
    std::string windowTitle = "Proces " + std::to_string(world_rank);

    std::cout << "[Proces " << world_rank << "/" << world_size << "] Rozpoczynam przetwarzanie: " << myVideoFile << std::endl;

    // ===============================================================
    // Reszta kodu jest identyczna jak poprzednio,
    // tylko używa zmiennej 'myVideoFile' zamiast stałej ścieżki
    // ===============================================================

    const int MOTION_THRESHOLD = 25;

    cv::VideoCapture cap(myVideoFile); // <-- ZMIANA: Używamy pliku przypisanego przez MPI

    if (!cap.isOpened())
    {
        std::cerr << "[Proces " << world_rank << "] BLAD: Nie mozna otworzyc pliku: " << myVideoFile << std::endl;
        MPI_Finalize(); // Zakończ MPI przed wyjściem
        return -1;
    }

    cv::Mat frame;
    cv::Mat grayFrame;
    cv::Mat motionMaskGPU;

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
            // std::cout << "[Proces " << world_rank << "] Koniec pliku, zapetlanie." << std::endl;
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            isFirstFrame = true;
            continue;
        }

        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        if (isFirstFrame)
        {
            width = grayFrame.cols;
            height = grayFrame.rows;
            // std::cout << "[Proces " << world_rank << "] Rozdzielczosc: " << width << "x" << height << std::endl;
            dataSize = width * height * sizeof(unsigned char);

            checkCudaError(cudaMalloc((void**)&d_current, dataSize), "cudaMalloc d_current");
            checkCudaError(cudaMalloc((void**)&d_prev, dataSize), "cudaMalloc d_prev");
            checkCudaError(cudaMalloc((void**)&d_mask_raw, dataSize), "cudaMalloc d_mask_raw");
            checkCudaError(cudaMalloc((void**)&d_mask_eroded, dataSize), "cudaMalloc d_mask_eroded");
            checkCudaError(cudaMemset(d_mask_eroded, 0, dataSize), "cudaMemset d_mask_eroded");
            motionMaskGPU.create(height, width, CV_8UC1);
            checkCudaError(cudaMemcpy(d_prev, grayFrame.data, dataSize, cudaMemcpyHostToDevice), "cudaMemcpy d_prev (first frame)");
            isFirstFrame = false;
            continue;
        }

        // === Cały potok GPU (bez zmian) ===
        checkCudaError(cudaMemcpy(d_current, grayFrame.data, dataSize, cudaMemcpyHostToDevice), "cudaMemcpy d_current");
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
        diffAndThresholdKernel << <numBlocks, threadsPerBlock >> > (d_mask_raw, d_current, d_prev, width, height, MOTION_THRESHOLD);
        erosionKernel << <numBlocks, threadsPerBlock >> > (d_mask_eroded, d_mask_raw, width, height);
        checkCudaError(cudaGetLastError(), "Kernel launch failure");
        checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        checkCudaError(cudaMemcpy(motionMaskGPU.data, d_mask_eroded, dataSize, cudaMemcpyDeviceToHost), "cudaMemcpy d_mask (D2H)");
        checkCudaError(cudaMemcpy(d_prev, d_current, dataSize, cudaMemcpyDeviceToDevice), "cudaMemcpy d_prev (D2D)");
        // === Koniec potoku GPU ===

        tm.stop();

        double fps = tm.getFPS();
        std::string fpsText = "FPS: " + std::to_string((int)fps);
        cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        // ZMIANA: Używamy unikalnych tytułów okien
        cv::imshow("Oryginal - " + windowTitle, frame);
        cv::imshow("Maska - " + windowTitle, motionMaskGPU);

        int key = cv::waitKey(1);
        if (key == 27) { // ESC
            break;
        }
        else if (key == 's' || key == 'S')
        {
            // ZMIANA: Zapisujemy z unikalną nazwą procesu
            std::string originalName = "proces_" + std::to_string(world_rank) + "_klatka_" + std::to_string(frameCounter) + "_oryginal.png";
            std::string maskName = "proces_" + std::to_string(world_rank) + "_klatka_" + std::to_string(frameCounter) + "_maska_ruch.png";

            cv::imwrite(originalName, frame);
            cv::imwrite(maskName, motionMaskGPU);

            std::cout << "[Proces " << world_rank << "] ZAPISANO klatki." << std::endl;
            frameCounter++;
        }
    }

    // --- Sprzątanie ---
    std::cout << "[Proces " << world_rank << "] Zamykanie..." << std::endl;
    cudaFree(d_current);
    cudaFree(d_prev);
    cudaFree(d_mask_raw);
    cudaFree(d_mask_eroded);
    cudaDeviceReset();

    cap.release();
    cv::destroyAllWindows();

    // === 4. FINALIZACJA MPI ===
    // Musi być na samym końcu, przed return
    MPI_Finalize();
    return 0;
}