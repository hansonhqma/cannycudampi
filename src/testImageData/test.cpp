#include <iostream>
#include <iomanip>
#include <fstream>

char* original = NULL;
char* result = NULL;
unsigned long long length = 0;
int width = 0;
int height = 0;

void readBytesFile(std::ifstream &input_file) {
    // Get the size of the file
    input_file.seekg(0, std::ios::end); // Move to the end of the file
    std::streampos file_size = input_file.tellg(); // Get the file size
    input_file.seekg(0, std::ios::beg); // Move back to the beginning of the file

    // Allocate memory to store the file data
    original = new char[file_size];
    result = new char[file_size];
    length = file_size;

    // Read the entire file into the allocated memory
    input_file.read(original, file_size);
}

void processImageName(const std::string& filename) {
    // Find the position of the first underscore
    size_t underscore_pos = filename.find('_');
    if (underscore_pos == std::string::npos) {
        std::cerr << "Invalid file name format." << std::endl;
        return;
    }

    // Find the position of the dot ('.')
    size_t dot_pos = filename.find('.');
    if (dot_pos == std::string::npos) {
        std::cerr << "Invalid file name format." << std::endl;
        return;
    }

    // Extract the width string (from the start to the first underscore)
    std::string width_str = filename.substr(0, underscore_pos);

    // Extract the height string (from the first underscore + 1 to the dot position)
    std::string height_str = filename.substr(underscore_pos + 1, dot_pos - underscore_pos - 1);

    // Convert width and height strings to integers
    try {
        width = std::stoi(width_str);
        height = std::stoi(height_str);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid width or height value in file name." << std::endl;
        return;
    }
}

void writeBytesFile(std::string filename) {
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos == std::string::npos) {
        std::cerr << "Invalid file name format." << std::endl;
        return;
    }
    filename.insert(dot_pos, "_res");

    std::ofstream output_file(filename, std::ios::binary);
    output_file.write(result, length);

    std::cout << "File saved as " << filename << std::endl;
}

void print_bytes() {
    // Iterate through the buffer and print each byte
    for (size_t i = 0; i < length; i++) {
        // Print each byte as a hexadecimal value (e.g., 0x7F)
        std::cout << "0x" << std::hex << (unsigned int)(unsigned char)original[i] << " ";
        
        // Optionally, print a newline after every 16 bytes for readability
        if ((i + 1) % 16 == 0) {
            std::cout << std::endl;
        }
    }
    
    // Print a final newline at the end for clarity
    std::cout << std::endl;
}

int main() {
    // Specify the file path
    std::string file_path = "32_32.dat"; // Change this to your file path

    // Open the file in binary mode for reading
    std::ifstream input_file(file_path, std::ios::binary);

    // Check if the file opened successfully
    if (!input_file) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return 1;
    }

    // extract the width and height
    processImageName(file_path);
    // initialize original and result arrays
    // read the bytes into original
    readBytesFile(input_file);
    result = original;
    //print_bytes();
    // write the result bytes to a file
    writeBytesFile(file_path);

    // Close the file
    input_file.close();

    delete[] original;
    delete[] result;

    return 0;
}
