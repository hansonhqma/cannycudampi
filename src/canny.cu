#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdio.h>

uint8_t *original;
uint8_t *result;

__device__ int Gx[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
__device__ int Gy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
__device__ double Gmax = 0;

uint64_t length = 0;
int width = 0;
int height = 0;

extern "C"
{
    bool CannyEdgeDetectionMPI(uint8_t *originalImage,
                        uint8_t *resultImage,
                        size_t imageWidth,
                        size_t imageHeight,
                        int threadsCount,
                        int rank);
}

__global__ void convolution_2d(uint8_t *&originalImage, const char maskOption, int *&resultImage, size_t width, size_t height)
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
                if ((start_r + i) < height) //&& (start_r + i) >= 0
                {
                    // Range check for columns
                    if ((start_c + j) < width) //&& (start_c + j) >= 0
                    {
                        // Accumulate result
                        temp += int((unsigned int)(unsigned char)(originalImage[(start_r + i) * width + (start_c + j)]) * mask[i * 3 + j]);
                    }
                }
            }
        }

        // Write back the result
        resultImage[index] = temp;
    }
}

__global__ void calculateGradiantAndAngles(double *&resultImage, double *&angles,
                                           int* &SobelX, int* &SobelY, size_t width, size_t height)
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

__global__ void normalizeG(double *&resultImage, size_t width, size_t height)
{
    size_t index = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + threadIdx.x;
    if (index < (uint64_t) width * (uint64_t) height) {
        resultImage[index] /= Gmax;
        resultImage[index] *= 255;
    }
}

__global__ void nonmaximumSuppression(double *&G, double *&angle, double *&suppressed, size_t width, size_t height) {
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

__global__ void doubleThreshold(double *&input, uint8_t *&thresh, int lowThresh, int highThresh, size_t width, size_t height) {
    size_t index = (uint64_t) blockIdx.x * (uint64_t) blockDim.x + threadIdx.x;

    if (index < (uint64_t) width * (uint64_t) height) {
        if (input[index] < lowThresh) {
            thresh[index] = 0;
        } else if (input[index] > highThresh) {
            thresh[index] = 255;
        } else {
            thresh[index] = (uint8_t) input[index];
        }
    }
}

__global__ void hysteresis(uint8_t *&input, uint8_t *&hyst, size_t width, size_t height) {
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

__global__ void writeResult(uint8_t *&resultImage, uint8_t *&calculations, size_t width, size_t height) {
    size_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < (uint64_t) width * height) {
        resultImage[index] = calculations[index];
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
    cudaSetDevice( rank );
    // Initialize helper memory
    uint8_t *original_image;
    uint8_t *result_image;
    int *SobelX;
    int *SobelY;
    double *G;
    double *angles;
    double *suppressed;
    uint8_t *thresh;
    uint8_t *hyst;
    cudaMallocManaged(&original_image, width * height * sizeof(uint8_t));
    cudaMallocManaged(&result_image, width * height * sizeof(uint8_t));
    cudaMemcpy(original_image, originalImage, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(result_image, resultImage, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMallocManaged(&SobelX, width * height * sizeof(int));
    cudaMallocManaged(&SobelY, width * height * sizeof(int));
    cudaMallocManaged(&G, width * height * sizeof(double));
    cudaMallocManaged(&angles, width * height * sizeof(double));
    cudaMallocManaged(&suppressed, width * height * sizeof(double));
    cudaMallocManaged(&thresh, width * height * sizeof(uint8_t));
    cudaMallocManaged(&hyst, width * height * sizeof(uint8_t));
    
    // determine num blocks by roughly dividing the array size by the number of threads
    dim3 blocks = dim3((imageWidth * imageHeight - 1 + threadsCount) / threadsCount, 1, 1);
    // three dimensional variable for number of threads
    dim3 threads = dim3(threadsCount, 1, 1);
    /*
    // calculate horizontal and vertical gradiant magnitude by convoluting the image with the kernel
    convolution_2d<<<blocks, threads>>>(original_image, 'x', SobelX, imageWidth, imageHeight);
    convolution_2d<<<blocks, threads>>>(original_image, 'y', SobelY, imageWidth, imageHeight);

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
    writeResult<<<blocks, threads>>>(result_image, hyst, imageWidth, imageHeight);
    cudaMemcpy(resultImage, result_image, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // for (uint64_t i=100; i<200; i++) {
    //     std::cout << angles[i] << " ";
    // }
    */
    // wait for all threads to finish and free helper memory
    cudaDeviceSynchronize();
    cudaFree(original_image);
    cudaFree(result_image);
    cudaFree(SobelX);
    cudaFree(SobelY);
    cudaFree(G);
    cudaFree(angles);
    cudaFree(suppressed);
    cudaFree(thresh);
    cudaFree(hyst);

    return true;
}

// helper function which lauches the kernel each iteration
bool CannyEdgeDetection(uint8_t *originalImage,
                        uint8_t *resultImage,
                        size_t imageWidth,
                        size_t imageHeight,
                        int threadsCount)
{
    // Initialize helper memory
    uint8_t *original_image;
    uint8_t *result_image;
    int *SobelX;
    int *SobelY;
    double *G;
    double *angles;
    double *suppressed;
    uint8_t *thresh;
    uint8_t *hyst;
    cudaMallocManaged(&original_image, width * height * sizeof(uint8_t));
    cudaMallocManaged(&result_image, width * height * sizeof(uint8_t));
    cudaMemcpy(original_image, originalImage, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(result_image, resultImage, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMallocManaged(&SobelX, width * height * sizeof(int));
    cudaMallocManaged(&SobelY, width * height * sizeof(int));
    cudaMallocManaged(&G, width * height * sizeof(double));
    cudaMallocManaged(&angles, width * height * sizeof(double));
    cudaMallocManaged(&suppressed, width * height * sizeof(double));
    cudaMallocManaged(&thresh, width * height * sizeof(uint8_t));
    cudaMallocManaged(&hyst, width * height * sizeof(uint8_t));

    // determine num blocks by roughly dividing the array size by the number of threads
    dim3 blocks = dim3((imageWidth * imageHeight - 1 + threadsCount) / threadsCount, 1, 1);
    // three dimensional variable for number of threads
    dim3 threads = dim3(threadsCount, 1, 1);

    // calculate horizontal and vertical gradiant magnitude by convoluting the image with the kernel
    convolution_2d<<<blocks, threads>>>(original_image, 'x', SobelX, imageWidth, imageHeight);
    convolution_2d<<<blocks, threads>>>(original_image, 'y', SobelY, imageWidth, imageHeight);

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
    writeResult<<<blocks, threads>>>(result_image, hyst, imageWidth, imageHeight);
    cudaMemcpy(resultImage, result_image, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // for (uint64_t i=100; i<200; i++) {
    //     std::cout << angles[i] << " ";
    // }

    // wait for all threads to finish and free helper memory
    cudaDeviceSynchronize();
    cudaFree(original_image);
    cudaFree(result_image);
    cudaFree(SobelX);
    cudaFree(SobelY);
    cudaFree(G);
    cudaFree(angles);
    cudaFree(suppressed);
    cudaFree(thresh);
    cudaFree(hyst);

    return true;
}

void readBytesFile(std::ifstream &input_file)
{
    // Get the size of the file
    input_file.seekg(0, std::ios::end);            // Move to the end of the file
    std::streampos file_size = input_file.tellg(); // Get the file size
    input_file.seekg(0, std::ios::beg);            // Move back to the beginning of the file

    // Allocate memory to store the file data
    length = file_size;
    original = new uint8_t[file_size];
    result = new uint8_t[file_size];

    // Read the entire file into the allocated memory
    input_file.read(reinterpret_cast<char *>(original), file_size);
}

void processImageName(const std::string &filename)
{
    // Find the position of the first underscore
    size_t underscore_pos = filename.find('_');
    if (underscore_pos == std::string::npos)
    {
        std::cerr << "Invalid file name format." << std::endl;
        return;
    }

    // Find the position of the dot ('.')
    size_t dot_pos = filename.find('.');
    if (dot_pos == std::string::npos)
    {
        std::cerr << "Invalid file name format." << std::endl;
        return;
    }

    // Extract the width string (from the start to the first underscore)
    std::string width_str = filename.substr(0, underscore_pos);

    // Extract the height string (from the first underscore + 1 to the dot position)
    std::string height_str = filename.substr(underscore_pos + 1, dot_pos - underscore_pos - 1);

    // Convert width and height strings to integers
    try
    {
        width = std::atoi(width_str.c_str());
        height = std::atoi(height_str.c_str());
    }
    catch (const std::invalid_argument &e)
    {
        std::cerr << "Invalid width or height value in file name." << std::endl;
        return;
    }
}

void writeBytesFile(std::string filename)
{
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos == std::string::npos)
    {
        std::cerr << "Invalid file name format." << std::endl;
        return;
    }
    filename.insert(dot_pos, "_res");

    std::ofstream output_file(filename, std::ios::binary);
    output_file.write(reinterpret_cast<const char *>(result), length);

    std::cout << "File saved as " << filename << std::endl;
}

void print_bytes()
{
    // Iterate through the buffer and print each byte
    for (size_t i = 0; i < length; i++)
    {
        // Print each byte as a hexadecimal value (e.g., 0x7F)
        // std::cout << "0x" << std::hex << (unsigned int)(unsigned char)original[i] << " ";
        std::cout << (unsigned int)(unsigned char)original[i] << " ";

        // Optionally, print a newline after every 16 bytes for readability
        if ((i + 1) % 16 == 0)
        {
            std::cout << std::endl;
        }
    }

    // Print a final newline at the end for clarity
    std::cout << std::endl;
}

// int main()
// {
//     // Specify the file path
//     std::string file_path = "32_32.dat"; // Change this to your file path

//     // Open the file in binary mode for reading
//     std::ifstream input_file(file_path, std::ios::binary);

//     // Check if the file opened successfully
//     if (!input_file)
//     {
//         std::cerr << "Failed to open file: " << file_path << std::endl;
//         return 1;
//     }

//     // extract the width and height
//     processImageName(file_path);
//     // initialize original and result arrays
//     // read the bytes into original
//     readBytesFile(input_file);
//     // perform Canny edge detection
//     CannyEdgeDetection(original, result, width, height, 1024);
//     //  write the result bytes to a file
//     writeBytesFile(file_path);

//     // Close the file
//     input_file.close();

//     // cudaFree(original);
//     // cudaFree(result);
//     delete[] original;
//     delete[] result;

//     return 0;
// }