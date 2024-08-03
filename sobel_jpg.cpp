#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <jpeglib.h>

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

RGBPixel img[IMAGE_HEIGHT][IMAGE_WIDTH];
GrayPixel grayscale[IMAGE_HEIGHT][IMAGE_WIDTH];
GrayPixel edges[IMAGE_HEIGHT][IMAGE_WIDTH];

void grayscaleConversion() {
    int x, y;
    for (y = 0; y < IMAGE_HEIGHT; y++) {
        for (x = 0; x < IMAGE_WIDTH; x++) {
            grayscale[y][x].gray = (uint8_t)((0.3 * img[y][x].red) +
                                              (0.59 * img[y][x].green) +
                                              (0.11 * img[y][x].blue));
        }
    }
}

void sobelEdgeDetection() {
    int gradient_x, gradient_y, x, y;
    int Gx[3][3] = {{-1, 0, 1},
                    {-2, 0, 2},
                    {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1},
                    { 0,  0,  0},
                    { 1,  2,  1}};

    for (y = 1; y < IMAGE_HEIGHT - 1; y++) {
        for (x = 1; x < IMAGE_WIDTH - 1; x++) {
            gradient_x = (Gx[0][0] * grayscale[y - 1][x - 1].gray) + (Gx[0][1] * grayscale[y - 1][x].gray) + (Gx[0][2] * grayscale[y - 1][x + 1].gray) +
                         (Gx[1][0] * grayscale[y][x - 1].gray) + (Gx[1][1] * grayscale[y][x].gray) + (Gx[1][2] * grayscale[y][x + 1].gray) +
                         (Gx[2][0] * grayscale[y + 1][x - 1].gray) + (Gx[2][1] * grayscale[y + 1][x].gray) + (Gx[2][2] * grayscale[y + 1][x + 1].gray);

            gradient_y = (Gy[0][0] * grayscale[y - 1][x - 1].gray) + (Gy[0][1] * grayscale[y - 1][x].gray) + (Gy[0][2] * grayscale[y - 1][x + 1].gray) +
                         (Gy[1][0] * grayscale[y][x - 1].gray) + (Gy[1][1] * grayscale[y][x].gray) + (Gy[1][2] * grayscale[y][x + 1].gray) +
                         (Gy[2][0] * grayscale[y + 1][x - 1].gray) + (Gy[2][1] * grayscale[y + 1][x].gray) + (Gy[2][2] * grayscale[y + 1][x + 1].gray);

            int gradient = abs(gradient_x) + abs(gradient_y);
            edges[y][x].gray = (uint8_t)(gradient > 128 ? 255 : gradient);
        }
    }
}

void loadJPEGImage(const char *filename) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    JSAMPARRAY buffer;
    int row_stride;

    FILE *infile = fopen(filename, "rb");
    if (!infile) {
        fprintf(stderr, "Error: Unable to open file %s for reading.\n", filename);
        exit(EXIT_FAILURE);
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    row_stride = cinfo.output_width * cinfo.output_components;
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);

    int x;
    while (cinfo.output_scanline < cinfo.output_height) {
        int y = cinfo.output_scanline;
        jpeg_read_scanlines(&cinfo, buffer, 1);
        for (x = 0; x < IMAGE_WIDTH; x++) {
            img[y][x].red = buffer[0][x * 3];
            img[y][x].green = buffer[0][x * 3 + 1];
            img[y][x].blue = buffer[0][x * 3 + 2];
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

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    if ((outfile = fopen(filename, "wb")) == NULL) {
        fprintf(stderr, "Error: Unable to open file %s for writing.\n", filename);
        exit(EXIT_FAILURE);
    }
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = IMAGE_WIDTH;
    cinfo.image_height = IMAGE_HEIGHT;
    cinfo.input_components = 1;
    cinfo.in_color_space = JCS_GRAYSCALE;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 100, TRUE);

    jpeg_start_compress(&cinfo, TRUE);

    row_stride = IMAGE_WIDTH * 1;
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = &image[cinfo.next_scanline][0].gray;
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
}

int main() {
    clock_t start, end;
    double cpu_time_used;

    loadJPEGImage("Large_image.jpg");
    start = clock();

    grayscaleConversion();

    

    sobelEdgeDetection();

    saveJPEGImage("Large_image_edge.jpg", edges);

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken for edge detection: %f seconds\n", cpu_time_used);

    return 0;
}
