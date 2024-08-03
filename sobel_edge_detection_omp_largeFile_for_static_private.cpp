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
#define MAX_THREADS 32  // Define the maximum expected number of threads

typedef struct {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
} RGBPixel;

typedef struct {
    uint8_t gray;
} GrayPixel;

// Declare static arrays for image data
static RGBPixel img[IMAGE_HEIGHT][IMAGE_WIDTH];
static GrayPixel grayscale[IMAGE_HEIGHT][IMAGE_WIDTH];
static GrayPixel edges[IMAGE_HEIGHT][IMAGE_WIDTH];

void grayscaleConversion() {
    #pragma omp parallel for
    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            grayscale[y][x].gray = (uint8_t)((0.3 * img[y][x].red) +
                                              (0.59 * img[y][x].green) +
                                              (0.11 * img[y][x].blue));
        }
    }
}



// Declare global static arrays for local edges processing
static GrayPixel local_edges_storage[MAX_THREADS][IMAGE_HEIGHT][IMAGE_WIDTH];

void sobelEdgeDetection() {
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        int rows_per_thread = IMAGE_HEIGHT / num_threads;  // Number of rows handled by each thread
        int start_row = thread_id * rows_per_thread;  // Starting row for this thread
        int end_row = (thread_id == num_threads - 1) ? IMAGE_HEIGHT : start_row + rows_per_thread;  // Last row handled by this thread

        // Pointer to the local edges for this thread
        GrayPixel (*local_edges)[IMAGE_WIDTH] = local_edges_storage[thread_id];

        #pragma omp for nowait
        for (int y = start_row + 1; y < end_row - 1; y++) {
            for (int x = 1; x < IMAGE_WIDTH - 1; x++) {
                int gradient_x = 0, gradient_y = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int ny = y + dy, nx = x + dx;
                        gradient_x += Gx[dy + 1][dx + 1] * grayscale[ny][nx].gray;
                        gradient_y += Gy[dy + 1][dx + 1] * grayscale[ny][nx].gray;
                    }
                }
                int gradient = abs(gradient_x) + abs(gradient_y);
                local_edges[y - start_row][x].gray = (uint8_t)(gradient > 255 ? 255 : gradient);
            }
        }

        // Merge local results back to the global edges array
        #pragma omp critical
        {
            for (int y = start_row + 1; y < end_row - 1; y++) {
                for (int x = 1; x < IMAGE_WIDTH - 1; x++) {
                    edges[y][x].gray = local_edges[y - start_row][x].gray;
                }
            }
        }
    }
}





void loadJPEGImage(const char *filename) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *infile;
    JSAMPARRAY buffer;
    int row_stride;

    if ((infile = fopen(filename, "rb")) == NULL) {
        fprintf(stderr, "Error: Unable to open file %s for reading.\n", filename);
        exit(EXIT_FAILURE);
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    if (cinfo.output_components != RGB_CHANNELS) {
        fprintf(stderr, "Error: JPEG must be in RGB format.\n");
        exit(EXIT_FAILURE);
    }

    row_stride = cinfo.output_width * cinfo.output_components;
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        for (int x = 0; x < cinfo.output_width; x++) {
            img[cinfo.output_scanline - 1][x].red = buffer[0][x * cinfo.output_components];
            img[cinfo.output_scanline - 1][x].green = buffer[0][x * cinfo.output_components + 1];
            img[cinfo.output_scanline - 1][x].blue = buffer[0][x * cinfo.output_components + 2];
        }
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
}

void saveJPEGImage(const char *filename, GrayPixel image[IMAGE_HEIGHT][IMAGE_WIDTH]) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *outfile;
    JSAMPROW row_pointer[1];
    int row_stride;

    if ((outfile = fopen(filename, "wb")) == NULL) {
        fprintf(stderr, "Error: Unable to open file %s for writing.\n", filename);
        exit(EXIT_FAILURE);
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = IMAGE_WIDTH;
    cinfo.image_height = IMAGE_HEIGHT;
    cinfo.input_components = 1;
    cinfo.in_color_space = JCS_GRAYSCALE;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 95, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = &image[cinfo.next_scanline][0].gray;
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
}

int main() {
    clock_t start, end;
    double cpu_time_used;
    printf("OpenMP version %d\n", _OPENMP);

    loadJPEGImage("Large_image.jpg");
    start = clock();
    grayscaleConversion();
    
    sobelEdgeDetection();
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken for edge detection: %f seconds\n", cpu_time_used);
    saveJPEGImage("Large_image_edge.jpg", edges);

    return 0;
}
