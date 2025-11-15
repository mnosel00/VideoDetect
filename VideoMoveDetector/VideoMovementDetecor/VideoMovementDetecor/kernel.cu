#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Nagłówki OpenCV
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>    
#include <opencv2/highgui.hpp>    
#include <opencv2/imgproc.hpp>    
#include <opencv2/core/utility.hpp> // Dla cv::TickMeter

// OpenMP już nie jest potrzebne
// #include <omp.h> 

#include <stdio.h>
#include <iostream>
#include <string> 

// ======================================================================
// KERNEL CUDA 1: Różnicowanie i Progowanie (bez zmian)
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
// NOWY KERNEL CUDA 2: Morfologia (Erozja)
// ======================================================================

/**
 * @brief Kernel CUDA do wykonania erozji 3x3.
 * Czyta z 'srcMask' (wynik kernela 1) i zapisuje do 'dstMask'.
 * Nie można bezpiecznie czytać i pisać do tej samej pamięci w morfologii,
 * dlatego potrzebujemy dwóch oddzielnych buforów.
 */
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
        exit(EXIT_FAILURE);
    }
}

// ======================================================================
// GŁÓWNA FUNKCJA PROGRAMU (ZMIANY)
// ======================================================================

int main()
{
    const int MOTION_THRESHOLD = 25;

    // ZMIANA TUTAJ: Podaj ścieżkę do swojego pliku wideo
    cv::VideoCapture cap("C:\\Users\\mnosel\\Downloads\\test0.mp4");
    if (!cap.isOpened())
    {
        std::cerr << "BLAD: Nie mozna otworzyc pliku wideo!" << std::endl;
        std::cin.get();
        return -1;
    }

    std::cout << "Otwarto plik wideo. Rozpoczynanie przetwarzania..." << std::endl;
    std::cout << "Nacisnij 'ESC', aby zakonczyc." << std::endl;
    std::cout << "Nacisnij 's', aby zapisac klatki." << std::endl;

    cv::Mat frame;
    cv::Mat grayFrame;
    cv::Mat motionMaskGPU;  // Zmieniamy nazwę, to będzie końcowy wynik z GPU

    // Wskaźniki do pamięci na GPU (Device)
    unsigned char* d_current = nullptr;
    unsigned char* d_prev = nullptr;
    unsigned char* d_mask_raw = nullptr;    // Bufor na surową maskę (wynik kernela 1)
    unsigned char* d_mask_eroded = nullptr; // Bufor na maskę po erozji (wynik kernela 2)

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
            std::cout << "Koniec pliku wideo. Zapetlanie..." << std::endl;
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            isFirstFrame = true;
            continue;
        }

        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        if (isFirstFrame)
        {
            width = grayFrame.cols;
            height = grayFrame.rows;
            std::cout << "Rozdzielczosc przetwarzania: " << width << "x" << height << std::endl;
            dataSize = width * height * sizeof(unsigned char);

            // Alokujemy 4 bufory na GPU
            checkCudaError(cudaMalloc((void**)&d_current, dataSize), "cudaMalloc d_current");
            checkCudaError(cudaMalloc((void**)&d_prev, dataSize), "cudaMalloc d_prev");
            checkCudaError(cudaMalloc((void**)&d_mask_raw, dataSize), "cudaMalloc d_mask_raw");
            checkCudaError(cudaMalloc((void**)&d_mask_eroded, dataSize), "cudaMalloc d_mask_eroded");

            // Zerujemy pamięć bufora wyjściowego (ważne dla ramki w erozji)
            checkCudaError(cudaMemset(d_mask_eroded, 0, dataSize), "cudaMemset d_mask_eroded");

            motionMaskGPU.create(height, width, CV_8UC1);

            checkCudaError(cudaMemcpy(d_prev, grayFrame.data, dataSize, cudaMemcpyHostToDevice), "cudaMemcpy d_prev (first frame)");
            isFirstFrame = false;
            continue;
        }

        //POCZĄTEK PRZETWARZANIA
        checkCudaError(cudaMemcpy(d_current, grayFrame.data, dataSize, cudaMemcpyHostToDevice), "cudaMemcpy d_current");

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // Wynik trafia do d_mask_raw
        diffAndThresholdKernel << <numBlocks, threadsPerBlock >> > (d_mask_raw, d_current, d_prev, width, height, MOTION_THRESHOLD);

        
        // Kernel czyta z d_mask_raw i zapisuje do d_mask_eroded
        erosionKernel << <numBlocks, threadsPerBlock >> > (d_mask_eroded, d_mask_raw, width, height);

        checkCudaError(cudaGetLastError(), "Kernel launch failure");
        checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        // 4. Kopiowanie wyniku  z GPU -> CPU
        checkCudaError(cudaMemcpy(motionMaskGPU.data, d_mask_eroded, dataSize, cudaMemcpyDeviceToHost), "cudaMemcpy d_mask (D2H)");

        // 6. Aktualizacja bufora d_prev (D2D)
        checkCudaError(cudaMemcpy(d_prev, d_current, dataSize, cudaMemcpyDeviceToDevice), "cudaMemcpy d_prev (D2D)");
        //KONIEC PRZETWARZANIA

        tm.stop();

        double fps = tm.getFPS();
        std::string fpsText = "FPS: " + std::to_string((int)fps);

        cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        // Wyświetlanie wyników
        cv::imshow("Oryginal (Wideo) z FPS", frame);
        cv::imshow("Maska Ruchu (z GPU, po erozji)", motionMaskGPU);
        // Usunęliśmy okno 'Maska Ruchu (z GPU)'

        int key = cv::waitKey(1);
        if (key == 27) { // ESC
            break;
        }
        // --- POPRAWIONA SEKCJA ZAPISYWANIA (3 KLATKI) ---
        else if (key == 's' || key == 'S')
        {
            // === USTAWIENIE ŚCIEŻKI ===
            // Zmień "mnosel" na swoją nazwę użytkownika!
            std::string savePath = "C:\\Users\\mnosel\\Desktop\\";

            // Nazwy plików dla wszystkich trzech obrazów
            std::string originalName = savePath + "klatka_" + std::to_string(frameCounter) + "_1_oryginal.png";
            std::string maskRawName = savePath + "klatka_" + std::to_string(frameCounter) + "_2_maska_z_szumem.png";
            std::string maskCleanName = savePath + "klatka_" + std::to_string(frameCounter) + "_3_maska_oczyszczona.png";

            // Sprawdzamy, czy obraz oryginalny nie jest pusty
            if (frame.empty()) {
                std::cerr << "BLAD: Proba zapisu pustej klatki!" << std::endl;
            }
            else
            {
                // ---- NOWY KROK DLA WERSJI FULL-GPU ----
                // Tworzymy tymczasowy kontener 'Mat' na CPU
                cv::Mat h_mask_raw(height, width, CV_8UC1);
                // Kopiujemy "brudną" maskę (z szumami) z d_mask_raw (GPU) do h_mask_raw (CPU)
                checkCudaError(cudaMemcpy(h_mask_raw.data, d_mask_raw, dataSize, cudaMemcpyDeviceToHost), "Copy raw mask D2H for saving");
                // -----------------------------------------

                // Zapisujemy wszystkie trzy obrazy
                bool success1 = cv::imwrite(originalName, frame);         // 1. Oryginał
                bool success2 = cv::imwrite(maskRawName, h_mask_raw);   // 2. Maska "z szumami" (którą właśnie skopiowaliśmy)
                bool success3 = cv::imwrite(maskCleanName, motionMaskGPU); // 3. Maska "oczyszczona" (jest już na CPU)

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
    cudaFree(d_mask_raw); // Sprzątamy nowy bufor
    cudaFree(d_mask_eroded); // Sprzątamy nowy bufor
    cudaDeviceReset();

    cap.release();
    cv::destroyAllWindows();

    return 0;
}