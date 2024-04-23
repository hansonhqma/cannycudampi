#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>

#include <mpi.h>

#define ROOT_PROC 0


typedef uint8_t PIXEL;

extern int CannyDriver(  PIXEL* originalImage_host,
                            PIXEL* resultImage_host,
                            size_t imageWidth,
                            size_t imageHeight,
                            int threadsCount );

int RANK, CLUSTER_SIZE;

size_t FRAME_HEIGHT, FRAME_WIDTH, FRAME_COUNT;


void increment_buffer(PIXEL* source, PIXEL* dest, size_t frame_size){
    for(int i=0;i<frame_size;++i){
        dest[i] = source[i] + 1;
    }
}


int main(int argc, char** argv){

    if(argc!=3){
        fprintf(stderr, "usage: ./a.out <rawdata> <thread_count>\n");
        exit(EXIT_FAILURE);
    }

    // need to copy filename into new buffer since token modifies original string
    char filename[32];
    strcpy(filename, argv[1]);

    int THREAD_COUNT = atoi(argv[2]);
    
    // get MPI world cluster size and proc number
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &CLUSTER_SIZE);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);

    // tokenize filename into video data
    const char *delim = "_";
    char* token = strtok(argv[1], delim);
    FRAME_HEIGHT = atoi(token);
    token = strtok(NULL, delim);
    FRAME_WIDTH = atoi(token);
    token = strtok(NULL, delim);
    FRAME_COUNT = atoi(token);

    // calculate reading offset
    size_t FRAME_SIZE = FRAME_HEIGHT * FRAME_WIDTH;
    size_t INITIAL_BYTE_OFFSET = RANK * FRAME_SIZE;
    size_t PROC_FRAME_OFFSET = FRAME_SIZE * CLUSTER_SIZE;
    //      'PROC_FRAME_OFFSET' is the number of bytes ahead a proc
    //      has to read in order to get its next frame

    if(RANK == ROOT_PROC){
        printf("Height: %zu, Width: %zu, Frame count: %zu\n", FRAME_HEIGHT, FRAME_WIDTH, FRAME_COUNT);
        printf("Frame size: %zu, PROC_FRAME_OFFSET: %zu\n", FRAME_SIZE, PROC_FRAME_OFFSET);
        printf("CUDA runtime loading with %d threads\n", THREAD_COUNT);
    }

    // open concurrent i/o for reading
    MPI_File SOURCE_FH;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &SOURCE_FH);

    // open concurrnet i/o for writing
    MPI_File DEST_FH;
    int status = MPI_File_open(MPI_COMM_WORLD, "DATA_OUT", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &DEST_FH);

    if(status){
        perror("Opening write file failed\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    
    // calculate necessary iterations for MPI cluster size
    
    size_t PROC_ITERATIONS = FRAME_COUNT / CLUSTER_SIZE; // gets base iter count
    // procs that need one more iteration are given that information
    if(RANK < (FRAME_COUNT % CLUSTER_SIZE)){ PROC_ITERATIONS++;}
    
    printf("PROC %d RUNNING %zu ITERATIONS - STARTING AT %zu BYTES\n", RANK, PROC_ITERATIONS, INITIAL_BYTE_OFFSET);
    PIXEL* frame_data = calloc(FRAME_SIZE, sizeof(PIXEL));
    PIXEL* result_data = calloc(FRAME_SIZE, sizeof(PIXEL));

    for(int iter = 0; iter < PROC_ITERATIONS; ++iter){
        // update byte offset
        size_t BYTE_OFFSET = INITIAL_BYTE_OFFSET + iter * PROC_FRAME_OFFSET;

        printf("Proc %d running iteration %d, reading at offset %zu\n", RANK, iter, BYTE_OFFSET);

        // grab frame data from file into array
        MPI_File_read_at(SOURCE_FH, BYTE_OFFSET, frame_data, FRAME_SIZE, MPI_BYTE, MPI_STATUS_IGNORE);

        printf("result ptr at %p\n", result_data);

        CannyDriver(frame_data, result_data, FRAME_WIDTH, FRAME_HEIGHT, THREAD_COUNT); 

        // write array back into dest file
        MPI_File_write_at(DEST_FH, BYTE_OFFSET, result_data, FRAME_SIZE, MPI_BYTE, MPI_STATUS_IGNORE);
       
        
    }

    MPI_File_close(&SOURCE_FH);
    MPI_File_close(&DEST_FH);

    free(frame_data);
    free(result_data);
    MPI_Finalize();
}

