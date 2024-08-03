#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <jpeglib.h>
#include <time.h>

#define IMAGE_WIDTH 30000
#define IMAGE_HEIGHT 22943
#define RGB_CHANNELS 3

typedef struct {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
} RGBPixel;

typedef struct {
    uint8_t gray;
} GrayPixel;

RGBPixel** img;
GrayPixel** grayscale;
GrayPixel** edges;

void allocateMemory() {
    img = (RGBPixel**) malloc(IMAGE_HEIGHT * sizeof(*img));
    grayscale = (GrayPixel**) malloc(IMAGE_HEIGHT * sizeof(*grayscale));
    edges = (GrayPixel**) malloc(IMAGE_HEIGHT * sizeof(*edges));
    
    for (int i = 0; i < IMAGE_HEIGHT; i++) {
        img[i] = (RGBPixel*) malloc(IMAGE_WIDTH * sizeof(**img));
        grayscale[i] = (GrayPixel*) malloc(IMAGE_WIDTH * sizeof(**grayscale));
        edges[i] = (GrayPixel*) malloc(IMAGE_WIDTH * sizeof(**edges));
    }
}

void freeMemory() {
    for (int i = 0; i < IMAGE_HEIGHT; i++) {
        free(img[i]);
        free(grayscale[i]);
        free(edges[i]);
    }
    free(img);
    free(grayscale);
    free(edges);
}

void grayscaleConversion() {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            grayscale[y][x].gray = (uint8_t)((0.3 * img[y][x].red) +
                                              (0.59 * img[y][x].green) +
                                              (0.11 * img[y][x].blue));
        }
    }
}

/*void sobelEdgeDetection() {
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0,  0,  0}, {1,  2,  1}};

    #pragma omp parallel for collapse(2)
    for (int y = 1; y < IMAGE_HEIGHT - 1; y++) {
        for (int x = 1; x < IMAGE_WIDTH - 1; x++) {
            int gradient_x = 0;
            int gradient_y = 0;

            
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    gradient_x += Gx[dy + 1][dx + 1] * grayscale[y + dy][x + dx].gray;
                    gradient_y += Gy[dy + 1][dx + 1] * grayscale[y + dy][x + dx].gray;
                }
            }

            int gradient = abs(gradient_x) + abs(gradient_y);
            edges[y][x].gray = (uint8_t)(gradient > 255 ? 255 : gradient);
        }
    }
}*/

/*void sobelEdgeDetection() {
    #pragma omp parallel
    {
        // Private buffers for each thread
        GrayPixel** private_edges = (GrayPixel**) malloc(IMAGE_HEIGHT * sizeof(*private_edges));
        for (int i = 0; i < IMAGE_HEIGHT; i++) {
            private_edges[i] = (GrayPixel*) calloc(IMAGE_WIDTH, sizeof(**private_edges));
        }

        #pragma omp for collapse(2)
        for (int y = 1; y < IMAGE_HEIGHT - 1; y++) {
            for (int x = 1; x < IMAGE_WIDTH - 1; x++) {
                int gradient_x = 0;
                int gradient_y = 0;
                int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
                int Gy[3][3] = {{-1, -2, -1}, {0,  0,  0}, {1,  2,  1}};
                
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        gradient_x += Gx[dy + 1][dx + 1] * grayscale[y + dy][x + dx].gray;
                        gradient_y += Gy[dy + 1][dx + 1] * grayscale[y + dy][x + dx].gray;
                    }
                }
                int gradient = abs(gradient_x) + abs(gradient_y);
                private_edges[y][x].gray = (uint8_t)(gradient > 255 ? 255 : gradient);
            }
        }

        // Merge results
        #pragma omp critical
        {
            for (int y = 1; y < IMAGE_HEIGHT - 1; y++) {
                for (int x = 1; x < IMAGE_WIDTH - 1; x++) {
                    edges[y][x].gray = private_edges[y][x].gray;
                }
            }
        }

        for (int i = 0; i < IMAGE_HEIGHT; i++) {
            free(private_edges[i]);
        }
        free(private_edges);
    }
}*/

void sobelEdgeDetection() {
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0,  0,  0}, {1,  2,  1}};

    #pragma omp parallel for collapse(2)
    for (int y = 1; y < IMAGE_HEIGHT - 1; y++) {
        for (int x = 1; x < IMAGE_WIDTH - 1; x++) {
            int gradient_x = 0;
            int gradient_y = 0;

            
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    gradient_x += Gx[dy + 1][dx + 1] * grayscale[y + dy][x + dx].gray;
                    gradient_y += Gy[dy + 1][dx + 1] * grayscale[y + dy][x + dx].gray;
                }
            }

            int gradient = abs(gradient_x) + abs(gradient_y);
            edges[y][x].gray = (uint8_t)(gradient > 255 ? 255 : gradient);
        }
    }
}

void loadJPEGImage(const char *filename) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *infile;
    JSAMPARRAY buffer;
    int row_stride;

    // Open the JPEG file
    if ((infile = fopen(filename, "rb")) == NULL) {
        fprintf(stderr, "Error: Unable to open file %s for reading.\n", filename);
        exit(EXIT_FAILURE);
    }

    // Set up the error handler
    cinfo.err = jpeg_std_error(&jerr);

    // Initialize the JPEG decompression object
    jpeg_create_decompress(&cinfo);

    // Specify the source of the data (the file)
    jpeg_stdio_src(&cinfo, infile);

    // Read the header to obtain file info
    jpeg_read_header(&cinfo, TRUE);

    // Start decompression
    jpeg_start_decompress(&cinfo);

    // Check to ensure the JPEG is in RGB format
    if (cinfo.output_components != RGB_CHANNELS) {
        fprintf(stderr, "Error: JPEG must be in RGB format.\n");
        exit(EXIT_FAILURE);
    }

    // Set row width in the buffer
    row_stride = cinfo.output_width * cinfo.output_components;

    // Allocate memory for one scanline
    buffer = (*cinfo.mem->alloc_sarray)
        ((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

    // Read the data
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        for (int x = 0; x < cinfo.output_width; x++) {
            img[cinfo.output_scanline - 1][x].red = buffer[0][x * cinfo.output_components];
            img[cinfo.output_scanline - 1][x].green = buffer[0][x * cinfo.output_components + 1];
            img[cinfo.output_scanline - 1][x].blue = buffer[0][x * cinfo.output_components + 2];
        }
    }

    // Finish decompression and close file
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
}


void saveJPEGImage(const char *filename, GrayPixel** image) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *outfile;
    JSAMPROW row_pointer[1];
    int row_stride;

    // Open file for writing
    if ((outfile = fopen(filename, "wb")) == NULL) {
        fprintf(stderr, "Error: Unable to open file %s for writing.\n", filename);
        exit(EXIT_FAILURE);
    }

    // Set up the error handler
    cinfo.err = jpeg_std_error(&jerr);

    // Initialize the JPEG compression object
    jpeg_create_compress(&cinfo);

    // Specify the destination of the data (the file)
    jpeg_stdio_dest(&cinfo, outfile);

    // Set parameters for the output file
    cinfo.image_width = IMAGE_WIDTH;
    cinfo.image_height = IMAGE_HEIGHT;
    cinfo.input_components = 1;
    cinfo.in_color_space = JCS_GRAYSCALE;

    // Set default compression parameters
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 95, TRUE);

    // Start compression
    jpeg_start_compress(&cinfo, TRUE);

    // Write pixel data
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = &image[cinfo.next_scanline][0].gray;
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    // Finish compression and close file
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
}




int main() {
    clock_t start, end;
    double cpu_time_used;
    printf("OpenMP version %d\n", _OPENMP);

    allocateMemory();
    loadJPEGImage("Large_image.jpg");
    start = clock();
    grayscaleConversion();
    
    sobelEdgeDetection();
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken for edge detection: %f seconds\n", cpu_time_used);
    saveJPEGImage("Large_image_edge.jpg", edges);
    freeMemory();

    

    return 0;
}

   
