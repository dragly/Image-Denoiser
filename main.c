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
        for(i = 1; i < inImage->height - 1; i++) {
            for(j = 1; j < inImage->width - 1; j++) {
                outImage->data[i][j] = inImage->data[i][j] + kappa * (inImage->data[i-1][j] + inImage->data[i][j-1]
                                                                      - 4 * inImage->data[i][j] + inImage->data[i][j+1] + inImage->data[i+1][j]);
            }
        }
        memcpy(inImage->data[0], outImage->data[0], inImage->height * inImage->width * sizeof(float));
        if(myRank > 0) {
            MPI_Send(inImage->data[1], inImage->width, MPI_FLOAT, myRank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(inImage->data[0], inImage->width, MPI_FLOAT, myRank - 1, 0, MPI_COMM_WORLD, &status);
        }
        if(myRank < numProcs - 1) {
            MPI_Send(inImage->data[inImage->height - 2], inImage->width, MPI_FLOAT, myRank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(inImage->data[inImage->height - 1], inImage->width, MPI_FLOAT, myRank + 1, 0, MPI_COMM_WORLD, &status);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        printf("Iteration %d\n",iteration);
        fflush(stdout);
    }
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
    float kappa = 0.1;
    int iterations = 100;
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

    // Divide the pixels evenly
    int myFirstPixel = (int)((myRank * mainImageHeight) / numProcs);
    int myLastPixel = (int)(((myRank + 1) * mainImageHeight) / numProcs - 1);
    if(myRank > 0) {
        myFirstPixel -= 1;
    }
    if(myRank < numProcs - 1) {
        myLastPixel += 1;
    }
    int myImageHeight = myLastPixel - myFirstPixel + 1;

    allocateImage(&inImage, myImageHeight, mainImageWidth);
    allocateImage(&outImage, myImageHeight, mainImageWidth);

    // Distribute the data from rank 0 to the children
    if(myRank == 0) {
        for(i = 1; i < numProcs; i++) {
            int rankFirstPixel = (int)((i * mainImageHeight) / numProcs);
            int rankLastPixel = (int)(((i + 1) * mainImageHeight) / numProcs - 1);
            if(i > 0) {
                rankFirstPixel -= 1;
            }
            if(i < numProcs - 1) {
                rankLastPixel += 1;
            }
            int rankImageHeight = rankLastPixel - rankFirstPixel + 1;
            MPI_Send(mainImage.data[rankFirstPixel], rankImageHeight * mainImageWidth, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }
        memcpy(inImage.data[0], mainImage.data[myFirstPixel], myImageHeight * mainImageWidth * sizeof(float));
    } else {
        MPI_Recv(inImage.data[0], myImageHeight * mainImageWidth, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
    }

    printf("So far so good %d\n", myRank);

    isoDiffusionDenoising(&inImage, &outImage, kappa, iterations, myRank, numProcs);

    // Distribute the data from rank 0 to the children
    if(myRank == 0) {
        for(i = 1; i < numProcs; i++) {
            int rankFirstPixel = (int)((i * mainImageHeight) / numProcs);
            int rankLastPixel = (int)(((i + 1) * mainImageHeight) / numProcs - 1);
            if(i > 0) {
                rankFirstPixel -= 1;
            }
            if(i < numProcs - 1) {
                rankLastPixel += 1;
            }
            int rankImageHeight = rankLastPixel - rankFirstPixel + 1;
            MPI_Recv(mainImage.data[rankFirstPixel], rankImageHeight * mainImageWidth, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
        }
        memcpy(mainImage.data[myFirstPixel], outImage.data[0], myImageHeight * mainImageWidth * sizeof(float));
    } else {
        int myFirstPixel = (int)((myRank * mainImageHeight) / numProcs);
        int myLastPixel = (int)(((myRank + 1) * mainImageHeight) / numProcs - 1);
        if(myRank > 0) {
            myFirstPixel -= 1;
        }
        if(myRank < numProcs - 1) {
            myLastPixel += 1;
        }
        int myImageHeight = myLastPixel - myFirstPixel + 1;
        MPI_Send(outImage.data[0], myImageHeight * mainImageWidth, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
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

