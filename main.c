#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "simplejpeg/import_export_jpeg.h"

typedef struct
{
    float **data;
    int height;
    int width;
} image;

void allocateImage(image *allocationImage, int imageHeight, int imageWidth) {
    // allocate space for image, making one big allocation for a continuous slab of memory
    float *dataDummy = (float*)malloc(imageHeight * imageWidth * sizeof(float));
    allocationImage->data = (float**)malloc(imageHeight * sizeof(float*));
    allocationImage->height = imageHeight;
    allocationImage->width = imageWidth;
    int i = 0;
    // point each row of the image to a point in dataDummy
    for(i = 0; i < imageHeight; i++) {
        allocationImage->data[i] = dataDummy + i * imageWidth;
    }
}

void deallocateImage(image *deallocationImage) {
    int i = 0;
    float *dataDummy = deallocationImage->data[i];
    // deallocate the list of rows
    free(deallocationImage->data);
    // deallocate the underlying 1D data
    free(dataDummy);
}

void isoDiffusionDenoising(image *inImage, image *outImage, float kappa, int iterations, int myRank, int numProcs) {
    MPI_Status status;
    int i;
    int j;
    int iteration;
    memcpy(outImage->data[0], inImage->data[0], inImage->height * inImage->width * sizeof(float));
    for(iteration = 0; iteration < iterations; iteration++) {
        // Corners
        i = 0;
        j = 0;
        outImage->data[i][j] = inImage->data[i][j] + kappa * (- 3 * inImage->data[i][j] + inImage->data[i][j+1] + inImage->data[i+1][j]);
        i = outImage->height - 1;
        j = 0;
        outImage->data[i][j] = inImage->data[i][j] + kappa * (- 3 * inImage->data[i][j] + inImage->data[i][j+1] + inImage->data[i-1][j]);
        i = outImage->height - 1;
        j = outImage->width - 1;
        outImage->data[i][j] = inImage->data[i][j] + kappa * (- 3 * inImage->data[i][j] + inImage->data[i][j-1] + inImage->data[i-1][j]);
        i = 0;
        j = outImage->width - 1;
        outImage->data[i][j] = inImage->data[i][j] + kappa * (- 3 * inImage->data[i][j] + inImage->data[i][j-1] + inImage->data[i+1][j]);
        // end Corners

        // Sides
        // i = 0
        i = 0;
        for(j = 1; j < inImage->width - 1; j++) {
            outImage->data[i][j] = inImage->data[i][j] + kappa * (inImage->data[i][j-1]
                                                                  - 3 * inImage->data[i][j] + inImage->data[i][j+1] + inImage->data[i+1][j]);
        }
        // i = max
        i = inImage->height - 1;
        for(j = 1; j < inImage->width - 1; j++) {
            outImage->data[i][j] = inImage->data[i][j] + kappa * (inImage->data[i][j-1]
                                                                  - 3 * inImage->data[i][j] + inImage->data[i][j+1] + inImage->data[i-1][j]);
        }
        // j = 0
        j = 0;
        for(i = 1; i < inImage->height - 1; i++) {
            outImage->data[i][j] = inImage->data[i][j] + kappa * (inImage->data[i-1][j]
                                                                  - 3 * inImage->data[i][j] + inImage->data[i][j+1] + inImage->data[i+1][j]);
        }
        // j = max
        j = inImage->width - 1;
        for(i = 1; i < inImage->height - 1; i++) {
            outImage->data[i][j] = inImage->data[i][j] + kappa * (inImage->data[i-1][j]
                                                                  - 3 * inImage->data[i][j] + inImage->data[i][j-1] + inImage->data[i+1][j]);
        }
        // end Sides

        // all the central pixels
        for(i = 1; i < inImage->height - 1; i++) {
            for(j = 1; j < inImage->width - 1; j++) {
                outImage->data[i][j] = inImage->data[i][j] + kappa * (inImage->data[i-1][j] + inImage->data[i][j-1]
                                                                      - 4 * inImage->data[i][j] + inImage->data[i][j+1] + inImage->data[i+1][j]);
            }
        }
        // copy the data back to the inImage matrix
        memcpy(inImage->data[0], outImage->data[0], inImage->height * inImage->width * sizeof(float));
        // send the next-to-boundary data to the neighbours' boundaries
        if(myRank > 0) {
            MPI_Send(outImage->data[1], inImage->width, MPI_FLOAT, myRank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(inImage->data[0], inImage->width, MPI_FLOAT, myRank - 1, 0, MPI_COMM_WORLD, &status);
        }
        if(myRank < numProcs - 1) {
            MPI_Send(outImage->data[inImage->height - 2], inImage->width, MPI_FLOAT, myRank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(inImage->data[inImage->height - 1], inImage->width, MPI_FLOAT, myRank + 1, 0, MPI_COMM_WORLD, &status);
        }
    }
}

void calculateMyPixels(int *myFirstPixel, int *myLastPixel, int *myImageHeight, int myRank, int numProcs, int mainImageHeight) {
    // Divide the pixels evenly
    *myFirstPixel = (int)((myRank * mainImageHeight) / numProcs);
    *myLastPixel = (int)(((myRank + 1) * mainImageHeight) / numProcs - 1);
    // Offset the inner parts by a pixel in each direction to cover all the inner boundaries
    if(myRank > 0) {
        *myFirstPixel -= 1;
    }
    if(myRank < numProcs - 1) {
        *myLastPixel += 1;
    }
    *myImageHeight = *myLastPixel - *myFirstPixel + 1; // +1 due to the fact that the difference in index value gives
}

int main(int argc, char** argv)
{
    int i;
    int j;
    int mainImageHeight = -1;
    int mainImageWidth = -1;
    image mainImage;
    image inImage;
    image outImage;
    int numComponents = -1;
    float kappa = 0.01;
    int iterations = 60;
    unsigned char *imageChars;
    char *inFileName = "noisy-paprika.jpg";
    char *outFileName = "smooth-paprika.jpg";
    MPI_Status status;

    // MPI variables
    int myRank;
    int numProcs;

    // Init MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    if(myRank == 0) {
        import_JPEG_file(inFileName, &imageChars, &mainImageHeight, &mainImageWidth, &numComponents);
        allocateImage(&mainImage, mainImageHeight, mainImageWidth);
        // load the data into the image
        for(i = 0; i < mainImage.height; i++) {
            for(j = 0; j < mainImage.width; j++) {
                mainImage.data[i][j] = (float)imageChars[i * mainImage.width + j];
            }
        }
    }

    // Distribute the image sizes
    MPI_Bcast(&mainImageHeight, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mainImageWidth, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int myFirstPixel = -1;
    int myLastPixel = -1;
    int myImageHeight = -1;
    calculateMyPixels(&myFirstPixel, &myLastPixel, &myImageHeight, myRank, numProcs, mainImageHeight);

    printf("%d %d %d\n", myFirstPixel, myLastPixel, myImageHeight);

    allocateImage(&inImage, myImageHeight, mainImageWidth);
    allocateImage(&outImage, myImageHeight, mainImageWidth);

    // Distribute the data from rank 0 to the children
    if(myRank == 0) {
        for(i = 1; i < numProcs; i++) {
            int rankFirstPixel = -1;
            int rankLastPixel = -1;
            int rankImageHeight = -1;
            calculateMyPixels(&rankFirstPixel, &rankLastPixel, &rankImageHeight, i, numProcs, mainImageHeight);
            MPI_Send(mainImage.data[rankFirstPixel], rankImageHeight * mainImageWidth, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }
        memcpy(inImage.data[0], mainImage.data[myFirstPixel], myImageHeight * mainImageWidth * sizeof(float));
    } else {
        MPI_Recv(inImage.data[0], myImageHeight * mainImageWidth, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
    }

    isoDiffusionDenoising(&inImage, &outImage, kappa, iterations, myRank, numProcs);

    // Send back all the data from the children to rank 0
    if(myRank == 0) {
        for(i = 1; i < numProcs; i++) {
            int rankFirstPixel = -1;
            int rankLastPixel = -1;
            int rankImageHeight = -1;
            calculateMyPixels(&rankFirstPixel, &rankLastPixel, &rankImageHeight, i, numProcs, mainImageHeight);
            MPI_Recv(mainImage.data[rankFirstPixel + 1], (rankImageHeight - 1) * mainImageWidth, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
        }
        memcpy(mainImage.data[myFirstPixel + 1], outImage.data[0], (myImageHeight - 1) * mainImageWidth * sizeof(float));
    } else {
        MPI_Send(outImage.data[1], (myImageHeight - 1) * mainImageWidth, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }

    // export the data of the new image
    if(myRank == 0) {
        // convert the data into chars
        for(i = 0; i < mainImage.height; i++) {
            for(j = 0; j < mainImage.width; j++) {
                imageChars[i * mainImage.width + j] = (char)(mainImage.data[i][j]);
            }
        }
        export_JPEG_file(outFileName, imageChars, mainImageHeight, mainImageWidth, numComponents, 100);
    }

    deallocateImage(&inImage);
    deallocateImage(&outImage);

    printf("Done %d\n", myRank);

    // Finalize application
    MPI_Finalize();
    return 0;
}

