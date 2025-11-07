#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Nagłówki OpenCV
// Nagłówki OpenCV
#include <opencv2/core.hpp>       // Podstawowe struktury (np. Mat)
#include <opencv2/imgcodecs.hpp>  // Do wczytywania/zapisywania obrazów (imread)
#include <opencv2/highgui.hpp>    // Do wyświetlania okien (imshow, waitKey)
#include <opencv2/imgproc.hpp>    // <-- DODAJ TĘ LINIĘ (przetwarzanie obrazów, np. cvtColor)

#include <stdio.h>
#include <iostream>

// Funkcja pomocnicza do sprawdzania błędów CUDA
void checkCudaError(cudaError_t status, const char* msg)
{
    if (status != cudaSuccess) {
        fprintf(stderr, "Blad CUDA: %s: %s\n", msg, cudaGetErrorString(status));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

/**
 * Kernel CUDA: Funkcja wykonywana na GPU.
 * Odwraca kolory obrazu (negatyw).
 * Obrazy OpenCV w C++ są domyślnie w formacie BGR.
 * uchar3 to typ wektorowy CUDA reprezentujący 3-kanałowy piksel (B, G, R).
 */
__global__ void invertKernel(uchar3* d_data, int width, int height)
{
    // Obliczanie unikalnego globalnego ID wątku dla obrazu 2D
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Upewniamy się, że wątek nie wychodzi poza granice obrazu
    if (x < width && y < height)
    {
        // Obliczamy liniowy indeks piksela
        int pixelIndex = y * width + x;

        // Pobieramy piksel
        uchar3 pixel = d_data[pixelIndex];

        // Odwracamy kolory (negatyw)
        pixel.x = 255 - pixel.x; // B
        pixel.y = 255 - pixel.y; // G
        pixel.z = 255 - pixel.z; // R

        // Zapisujemy przetworzony piksel z powrotem do pamięci GPU
        d_data[pixelIndex] = pixel;
    }
}

int main()
{
    // --------------------------------------------------------------------
    // ZMIEŃ TĘ ŚCIEŻKĘ na ścieżkę do dowolnego obrazu .jpg lub .png na Twoim dysku
    // WAŻNE: Używaj podwójnych ukośników wstecznych (\\) w ścieżce!
    std::string imagePath = "C:\\Users\\mnosel\\Downloads\\bursztynowa.jpg";
    // --------------------------------------------------------------------


    // 1. Wczytanie obrazu za pomocą OpenCV (na CPU)
    cv::Mat h_image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (h_image.empty())
    {
        std::cerr << "Nie mozna wczytac obrazu ze sciezki: " << imagePath << std::endl;
        std::cin.get();
        return -1;
    }

    // Konwertujemy obraz na 3 kanały BGR, jeśli jest inny (np. z alfą)
    if (h_image.channels() == 4) {
        cv::cvtColor(h_image, h_image, cv::COLOR_BGRA2BGR);
    }
    else if (h_image.channels() == 1) {
        cv::cvtColor(h_image, h_image, cv::COLOR_GRAY2BGR);
    }

    // Tworzymy kopię obrazu wyjściowego (też na CPU)
    cv::Mat h_image_out = h_image.clone();

    int width = h_image.cols;
    int height = h_image.rows;
    size_t dataSize = width * height * sizeof(uchar3);

    std::cout << "Obraz wczytany (" << width << "x" << height << "). Rozpoczynanie przetwarzania CUDA..." << std::endl;

    // 2. Alokacja pamięci na GPU
    uchar3* d_image_data = nullptr;
    checkCudaError(cudaMalloc((void**)&d_image_data, dataSize), "cudaMalloc");

    // 3. Kopiowanie danych obrazu z CPU (h_image.data) -> GPU (d_image_data)
    // h_image.data to wskaźnik na surowe dane pikseli
    checkCudaError(cudaMemcpy(d_image_data, h_image.data, dataSize, cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    // 4. Konfiguracja i uruchomienie Kernela CUDA
    // Definiujemy rozmiar bloku (np. 16x16 wątków na blok)
    dim3 threadsPerBlock(16, 16);

    // Obliczamy liczbę bloków potrzebną do pokrycia całego obrazu
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    invertKernel << <numBlocks, threadsPerBlock >> > (d_image_data, width, height);

    checkCudaError(cudaGetLastError(), "invertKernel launch");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // 5. Kopiowanie wyniku z GPU -> CPU (do h_image_out.data)
    checkCudaError(cudaMemcpy(h_image_out.data, d_image_data, dataSize, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    std::cout << "Przetwarzanie GPU zakonczone." << std::endl;

    // 6. Zwolnienie pamięci na GPU
    cudaFree(d_image_data);
    cudaDeviceReset();

    // 7. Wyświetlenie wyników za pomocą OpenCV
    cv::imshow("Oryginal (CPU)", h_image);
    cv::imshow("Obraz przetworzony (GPU - CUDA)", h_image_out);

    std::cout << "Nacisnij dowolny klawisz w oknie obrazu, aby zakonczyc..." << std::endl;
    cv::waitKey(0); // Czekaj na naciśnięcie klawisza

    return 0;
}