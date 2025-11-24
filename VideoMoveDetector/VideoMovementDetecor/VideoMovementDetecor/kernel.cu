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
#include <vector>
#include <mpi.h>

// --- KERNELE (BEZ ZMIAN) ---
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
        fprintf(stderr, "CUDA Error: %s\n", msg);
        MPI_Abort(MPI_COMM_WORLD, status);
    }
}

int main(int argc, char* argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<std::string> files = {
        "C:\\Users\\mnosel\\Downloads\\test0.mp4",
        "C:\\Users\\mnosel\\Downloads\\test1.mp4",
        "C:\\Users\\mnosel\\Downloads\\test2.mp4",
        "C:\\Users\\mnosel\\Downloads\\test3.mp4"
    };

    std::string myFile = files[rank % files.size()];

    // SIATKA OKIEN
    const int DISPLAY_W = 480;
    int displayH = 270;
    std::string titleOrig = "Orig P" + std::to_string(rank);
    std::string titleMask = "Mask P" + std::to_string(rank);

    cv::namedWindow(titleOrig, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(titleMask, cv::WINDOW_AUTOSIZE);

    cv::VideoCapture cap(myFile);
    if (!cap.isOpened()) { MPI_Finalize(); return -1; }

    cv::Mat frame, gray, maskGPU, smallF, smallM;
    unsigned char* d_curr, * d_prev, * d_raw, * d_eroded;

    // Inicjalizacja
    cap.read(frame);
    int w = frame.cols;
    int h = frame.rows;
    size_t sz = w * h;

    cudaMalloc((void**)&d_curr, sz); cudaMalloc((void**)&d_prev, sz);
    cudaMalloc((void**)&d_raw, sz); cudaMalloc((void**)&d_eroded, sz);
    cudaMemset(d_eroded, 0, sz);
    maskGPU.create(h, w, CV_8UC1);

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cudaMemcpy(d_prev, gray.data, sz, cudaMemcpyHostToDevice);

    // Ustawienie okien
    displayH = (int)((double)h / w * DISPLAY_W);
    int gx = rank % 2; int gy = rank / 2;
    cv::moveWindow(titleOrig, gx * (DISPLAY_W + 10), gy * (displayH + 40));
    cv::moveWindow(titleMask, gx * (DISPLAY_W + 10) + DISPLAY_W + 10, gy * (displayH + 40));

    cv::TickMeter tm;
    tm.start();
    int framesAvg = 0;
    int frameCounter = 0;

    while (true) {
        cap.read(frame);
        if (frame.empty()) { cap.set(cv::CAP_PROP_POS_FRAMES, 0); continue; }

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cudaMemcpy(d_curr, gray.data, sz, cudaMemcpyHostToDevice);

        dim3 th(16, 16);
        dim3 bl((w + 15) / 16, (h + 15) / 16);

        diffAndThresholdKernel << <bl, th >> > (d_raw, d_curr, d_prev, w, h, 25);
        erosionKernel << <bl, th >> > (d_eroded, d_raw, w, h);
        cudaDeviceSynchronize();

        cudaMemcpy(maskGPU.data, d_eroded, sz, cudaMemcpyDeviceToHost);
        cudaMemcpy(d_prev, d_curr, sz, cudaMemcpyDeviceToDevice);

        // --- WYPISYWANIE DANYCH DO WYKRESU ---
        framesAvg++;
        if (framesAvg == 100) {
            tm.stop();
            double fps = 100.0 / tm.getTimeSec();
            // Tylko proces 0 wypisuje, lub każdy
            // Tutaj każdy wypisuje, żeby sprawdzic czy są równe.
            std::cout << "[Proces " << rank << "] SREDNI FPS (MPI): " << fps << std::endl;
            tm.reset(); tm.start();
            framesAvg = 0;
        }

        // Wyświetlanie pomniejszone
        cv::resize(frame, smallF, cv::Size(DISPLAY_W, displayH));
        cv::resize(maskGPU, smallM, cv::Size(DISPLAY_W, displayH), 0, 0, cv::INTER_NEAREST);
        cv::imshow(titleOrig, smallF);
        cv::imshow(titleMask, smallM);

        // Obsługa klawiszy i ZAPIS
        int key = cv::waitKey(1);
        int l_save = (key == 's' || key == 'S') ? 1 : 0;
        int l_quit = (key == 27) ? 1 : 0;
        int g_save = 0, g_quit = 0;

        MPI_Allreduce(&l_save, &g_save, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&l_quit, &g_quit, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (g_quit) break;
        if (g_save) {
            std::string path = "C:\\Users\\mnosel\\Desktop\\";
            std::string name = "mpi_p" + std::to_string(rank) + "_" + std::to_string(frameCounter);

            // Pobranie brudnej maski
            cv::Mat h_raw(h, w, CV_8UC1);
            cudaMemcpy(h_raw.data, d_raw, sz, cudaMemcpyDeviceToHost);

            cv::imwrite(path + name + "_1_orig.png", frame);
            cv::imwrite(path + name + "_2_szum.png", h_raw);
            cv::imwrite(path + name + "_3_czyste.png", maskGPU);

            if (rank == 0) std::cout << "Zapisano zdjecia!" << std::endl;
            frameCounter++;
        }
    }

    cudaFree(d_curr); cudaFree(d_prev); cudaFree(d_raw); cudaFree(d_eroded);
    MPI_Finalize();
    return 0;
}