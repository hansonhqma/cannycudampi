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
    bool CannyEdgeDetection(uint8_t *originalImage,
                        uint8_t *resultImage,
                        size_t imageWidth,
                        size_t imageHeight,
                        int threadsCount);
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
    size_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t col = index % width;
    size_t row = index / width;
    if (index < (uint64_t) width * height)
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
    size_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < (uint64_t) width * height) {
        double magnitude = pow(pow(SobelX[index], 2) + pow(SobelY[index], 2), 0.5);
        resultImage[index] = magnitude;
        if (!Gmax || magnitude > Gmax) {
            Gmax = magnitude;
        }
        angles[index] = atan2(SobelY[index], SobelX[index]);
    }
    __syncthreads();
}

__global__ void normalizeG(double *&resultImage, size_t width, size_t height)
{
    size_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < (uint64_t) width * height) {
        resultImage[index] /= Gmax;
        resultImage[index] *= 255;
    }
}

__global__ void writeResult(uint8_t *&resultImage, double *&calculations, size_t width, size_t height) {
    size_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < (uint64_t) width * height) {
        resultImage[index] = (uint8_t)calculations[index];
    }
}

// helper function which lauches the kernel each iteration
bool CannyEdgeDetection(uint8_t *originalImage,
                        uint8_t *resultImage,
                        size_t imageWidth,
                        size_t imageHeight,
                        int threadsCount)
{
    // Initialize helper memory
    int *SobelX;
    int *SobelY;
    double *G;
    double *angles;
    cudaMallocManaged(&SobelX, width * height * sizeof(int));
    cudaMallocManaged(&SobelY, width * height * sizeof(int));
    cudaMallocManaged(&G, width * height * sizeof(double));
    cudaMallocManaged(&angles, width * height * sizeof(double));
    // determine num blocks by roughly dividing the array size by the number of threads
    dim3 blocks = dim3((imageWidth * imageHeight - 1 + threadsCount) / threadsCount, 1, 1);
    // three dimensional variable for number of threads
    dim3 threads = dim3(threadsCount, 1, 1);
    // calculate horizontal and vertical gradiant magnitude by convoluting the image with the kernel
    convolution_2d<<<blocks, threads>>>(originalImage, 'x', SobelX, imageWidth, imageHeight);
    convolution_2d<<<blocks, threads>>>(originalImage, 'y', SobelY, imageWidth, imageHeight);
    calculateGradiantAndAngles<<<blocks, threads>>>(G, angles, SobelX, SobelY, imageWidth, imageHeight);
    normalizeG<<<blocks, threads>>>(G, imageWidth, imageHeight);
    writeResult<<<blocks, threads>>>(resultImage, G, imageWidth, imageHeight);
    cudaDeviceSynchronize();
    cudaFree(SobelX);
    cudaFree(SobelY);
    cudaFree(G);
    cudaFree(angles);

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
    cudaMallocManaged(&original, length * sizeof(uint8_t));
    cudaMallocManaged(&result, length * sizeof(uint8_t));    

    // Read the entire file into the allocated memory
    input_file.read(reinterpret_cast<char *>(original), file_size);

    for (uint64_t i = 0; i < file_size; i++)
    {
        result[i] = 0;
    }
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

int main()
{
    // Specify the file path
    std::string file_path = "2340_1755.dat"; // Change this to your file path

    // Open the file in binary mode for reading
    std::ifstream input_file(file_path, std::ios::binary);

    // Check if the file opened successfully
    if (!input_file)
    {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return 1;
    }

    // extract the width and height
    processImageName(file_path);
    // initialize original and result arrays
    // read the bytes into original
    readBytesFile(input_file);
    // perform Canny edge detection
    CannyEdgeDetection(original, result, width, height, 1024);
    //  write the result bytes to a file
    writeBytesFile(file_path);

    // Close the file
    input_file.close();

    cudaFree(original);
    cudaFree(result);

    return 0;
}