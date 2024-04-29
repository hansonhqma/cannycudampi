#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdio.h>

__device__ int Gx[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
__device__ int Gy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
__device__ double Gmax = 0;

extern "C"
{
    bool CannyEdgeDetectionMPI(uint8_t *originalImage,
                        uint8_t *resultImage,
                        size_t imageWidth,
                        size_t imageHeight,
                        int threadsCount,
                        int rank);
}

__global__ void convolution_2d(uint8_t *originalImage, const char maskOption, int *resultImage, size_t width, size_t height)
{
    int* mask;
    if (maskOption == 'x') {
        mask = Gx;
    } else {
        mask = Gy;
    }
    // Calculate the global thread positions
    size_t index = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + threadIdx.x;
    size_t col = index % width;
    size_t row = index / width;
    if (index < (uint64_t) width * (uint64_t) height)
    {
        // Starting index for calculation
        size_t start_r = row - 1;
        size_t start_c = col - 1;

        // Temp value for accumulating the result
        int temp = 0;

        // Iterate over all the rows
        for (int i = 0; i < 3; i++)
        {
            // Go over each column
            for (int j = 0; j < 3; j++)
            {
                // Range check for rows
                if ((start_r + i) < height)
                {
                    // Range check for columns
                    if ((start_c + j) < width)
                    {
                        // Accumulate result
                        int num = originalImage[(start_r + i) * width + (start_c + j)];
                        temp += num * mask[i * 3 + j];
                    }
                }
            }
        }

        // Write back the result
        resultImage[index] = temp;
    }
}

__global__ void calculateGradiantAndAngles(double *resultImage, double *angles,
                                           int *SobelX, int *SobelY, size_t width, size_t height)
{
    size_t index = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + threadIdx.x;
    if (index < (uint64_t) width * (uint64_t) height) {
        double magnitude = pow(pow(SobelX[index], 2) + pow(SobelY[index], 2), 0.5);
        if (!Gmax || magnitude > Gmax) {
            Gmax = magnitude;
        }
        resultImage[index] = magnitude;
        angles[index] = atan2(SobelY[index], SobelX[index]) * 180 / 3.14;
        if (angles[index] < 0)
            angles[index] += 180;
    }
    __syncthreads();
}

__global__ void normalizeG(double *resultImage, size_t width, size_t height)
{
    size_t index = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + threadIdx.x;
    if (index < (uint64_t) width * (uint64_t) height) {
        resultImage[index] /= Gmax;
        resultImage[index] *= 255;
    }
}

__global__ void nonmaximumSuppression(double *G, double *angle, double *suppressed, size_t width, size_t height) {
    size_t index = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + threadIdx.x;
    // calculate x and y values for 2D representation
    uint64_t x = index % width;
    uint64_t y = index / width;

    if (index < (uint64_t) width * (uint64_t) height) {
        if (x >= 1 && x < width-1 && y >= 1 && y < height-1) {
            // values for adjacent pixels
            uint64_t y0 = ((y + height - 1) % height) * width;
            uint64_t y2 = ((y + 1) % height) * width;
            uint64_t x0 = (x + width - 1) % width;
            uint64_t x2 = (x + 1) % width;
            
            double adjacent1 = 255;
            double adjacent2 = 255;
            
            // Case 1: comparing left and right pixels
            if ((0 <= angle[index] && angle[index] < 22.5) || (157.5 <= angle[index] && angle[index] <= 180)) {
                adjacent1 = G[index-1];
                adjacent2 = G[index+1];
            } 
            
            // Case 2: comparing bottom right and top left
            else if ((22.5 <= angle[index] && angle[index] < 67.5)) {
                adjacent1 = G[x2 + y0];
                adjacent2 = G[x0 + y2];
            }
            
            // Case 3: comparing top and bottom
            else if ((67.5 <= angle[index] && angle[index] < 112.5)) {
                adjacent1 = G[x + y0];
                adjacent2 = G[x + y2];
            }

            // Case 4: comparing top right and bottom left
            else if ((112.5 <= angle[index] && angle[index] < 157.5)) {
                adjacent1 = G[x2 + y2];
                adjacent2 = G[x0 + y0];
            }

            // If the current pixel's change is the most extreme relative to its neighbors keep
            // Otherwise "suppress" the pixel by setting it to 0
            if ((G[index] < adjacent1) || (G[index] < adjacent2))
                suppressed[index] = 0;
            else
                suppressed[index] = G[index];
        }
    }
}

__global__ void doubleThreshold(double *input, int *thresh, int lowThresh, int highThresh, size_t width, size_t height) {
    size_t index = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + threadIdx.x;

    if (index < (uint64_t) width * (uint64_t) height) {
        if (input[index] < lowThresh) {
            thresh[index] = 0;
        } else if (input[index] > highThresh) {
            thresh[index] = 255;
        } else {
            thresh[index] = input[index];
        }
    }
}

__global__ void hysteresis(int *input, int *hyst, size_t width, size_t height) {
    size_t index = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + threadIdx.x;
    uint64_t x = index % width;
    uint64_t y = index / width;

    if (index < (uint64_t) width * (uint64_t) height) {
        if (x >= 1 && x < width-1 && y >= 1 && y < height-1) {
            if (input[index] != 0 && input[index] != 255) {
                uint64_t y0 = ((y + height - 1) % height) * width;
                uint64_t y2 = ((y + 1) % height) * width;
                uint64_t x0 = (x + width - 1) % width;
                uint64_t x2 = (x + 1) % width;

                if (input[x0 + y0] == 255 || input[x + y0] == 255 || input[x2 + y0] == 255 || input[x0 + y] == 255
                    || input[x2 + y] == 255 || input[x0 + y2] == 255 || input[x + y2] == 255 || input[x2 + y2] == 255 ) {
                        hyst[index] = 255;
                } else {
                    hyst[index] = 0;
                }
            } else {
                hyst[index] = input[index];
            }
        }
    }
}

__global__ void writeResult(uint8_t *resultImage, int *calculations, size_t width, size_t height) {
    size_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < (uint64_t) width * height) {
        resultImage[index] = (uint8_t) calculations[index];
    }
}

// helper function which lauches the kernel each iteration
bool CannyEdgeDetectionMPI(uint8_t *originalImage,
                        uint8_t *resultImage,
                        size_t imageWidth,
                        size_t imageHeight,
                        int threadsCount,
                        int rank)

{

    int availableCudaDevices;

    cudaGetDeviceCount(&availableCudaDevices);

    int selected_cuda_device = rank % availableCudaDevices;

    printf("rank %d selecting device %d\n", rank, selected_cuda_device);

    cudaError_t err = cudaSetDevice(selected_cuda_device);

    if(err){
        perror("cudaSetDevice failed");
    }

    // Initialize helper memory
    int *SobelX;
    int *SobelY;
    double *G;
    double *angles;
    double *suppressed;
    int *thresh;
    int *hyst;

    cudaMallocManaged(&SobelX, sizeof(int) * (uint64_t) imageWidth * (uint64_t) imageHeight);
    cudaMallocManaged(&SobelY, sizeof(int) * (uint64_t) imageWidth * (uint64_t) imageHeight);
    cudaMallocManaged(&G, sizeof(double) * (uint64_t) imageWidth * (uint64_t) imageHeight);
    cudaMallocManaged(&angles, sizeof(double) * (uint64_t) imageWidth * (uint64_t) imageHeight);
    cudaMallocManaged(&suppressed, sizeof(double) * (uint64_t) imageWidth * (uint64_t) imageHeight);
    cudaMallocManaged(&thresh, sizeof(int) * (uint64_t) imageWidth * (uint64_t) imageHeight);
    cudaMallocManaged(&hyst, sizeof(int) * (uint64_t) imageWidth * (uint64_t) imageHeight);
    // int *SobelX = new int[(uint64_t) imageWidth * (uint64_t) imageHeight];
    // int *SobelY = new int[(uint64_t) imageWidth * (uint64_t) imageHeight];
    // double *G = new double[(uint64_t) imageWidth * (uint64_t) imageHeight];
    // double *angles = new double[(uint64_t) imageWidth * (uint64_t) imageHeight];
    // double *suppressed = new double[(uint64_t) imageWidth * (uint64_t) imageHeight];
    // int *thresh = new int[(uint64_t) imageWidth * (uint64_t) imageHeight];
    // int *hyst = new int[(uint64_t) imageWidth * (uint64_t) imageHeight];
    
    // determine num blocks by roughly dividing the array size by the number of threads
    dim3 blocks = dim3((imageWidth * imageHeight - 1 + threadsCount) / threadsCount, 1, 1);
    // three dimensional variable for number of threads
    dim3 threads = dim3(threadsCount, 1, 1);

    
    // calculate horizontal and vertical gradiant magnitude by convoluting the image with the kernel
    convolution_2d<<<blocks, threads>>>(originalImage, 'x', SobelX, imageWidth, imageHeight);
    convolution_2d<<<blocks, threads>>>(originalImage, 'y', SobelY, imageWidth, imageHeight);

    // calculate grandient for edge detection and angles for nonmax suppression
    calculateGradiantAndAngles<<<blocks, threads>>>(G, angles, SobelX, SobelY, imageWidth, imageHeight);
    
    // normalize the gradients to keep the range between 0 and 255
    normalizeG<<<blocks, threads>>>(G, imageWidth, imageHeight);

    // perform nonmaximum suppression to reduce thickness of edges
    nonmaximumSuppression<<<blocks, threads>>>(G, angles, suppressed, imageWidth, imageHeight);

    // perform double thresholding which sets edges above the high threshold to 255,
    // edges below the low threshold to 0, above the high threshold are 255 and leaves the rest alone
    // the ones left alone are considered "weak edges"
    doubleThreshold<<<blocks, threads>>>(suppressed, thresh, 50, 128, imageWidth, imageHeight);
    
    // use the threshold array to perform hysteresis
    // analyze the weak edges and make them strong edges if they boarder a strong edge
    // otherwise set that pixel to 0
    hysteresis<<<blocks, threads>>>(thresh, hyst, imageWidth, imageHeight);
    // write result to resultImage for postprocessing
    writeResult<<<blocks, threads>>>(resultImage, hyst, imageWidth, imageHeight);

    // wait for all threads to finish and free helper memory
    cudaDeviceSynchronize();
    cudaFree(SobelX);
    cudaFree(SobelY);
    cudaFree(G);
    cudaFree(angles);
    cudaFree(suppressed);
    cudaFree(thresh);
    cudaFree(hyst);

    return true;
}
