#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

double* perform_relaxation(double** matrix, int size, double precision) {
    int rowSize = size - 2; // Accounting for the two borders on each side.
    if (rowSize < 1) { // At least 1 row that can be computed should exist.
        return NULL;
    }
    int processorCount;
    MPI_Comm_size(MPI_COMM_WORLD, &processorCount);

    int remainder, portion;
    remainder = rowSize % processorCount;
    portion = rowSize / processorCount;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rowSize < processorCount) { // At most 1 row per processor
        if (!rank)
            printf("Number of rows to be computed (%d) is smaller than the number of"
                   " processors (%d).\nEither increase the matrix size or "
                   "decrease the number of processors.\n", rowSize, processorCount);
        return NULL;
    }

    int concurrent = processorCount > 1; // 0 if sequential: processorCount = 1

    int start;
    if (rank < remainder) { // Spread remainder evenly across processors
        portion++; // Each processor gets an extra row to deal with.
        start = portion*rank;
    }
    else {
        start = (portion+1)*remainder + portion*(rank-remainder);
    }

    /* Allocate memory for respective portion of matrix */
    double** uniqueMatrix;
    // Portion plus above and below row
    uniqueMatrix = (double **) malloc((size_t) (portion + 2) * sizeof(double*));
    if (uniqueMatrix == NULL) {
        MPI_Abort(MPI_COMM_WORLD, -1);
        return NULL;
    }
    for (int i = 0; i < portion + 2; i++) {
        uniqueMatrix[i] = (double *) malloc((size_t) size * sizeof(double));
        if (uniqueMatrix[i] == NULL) {
            MPI_Abort(MPI_COMM_WORLD, -1);
            return NULL;
        }
    }

    /* Copy the portion on to the newly assigned memory space */
    for (int i = 0; i < portion+2; i++) {
        for (int j = 0; j < size; j++) {
            uniqueMatrix[i][j] = matrix[i+start][j];
        }
    }

    for (int i = 0; i < size; i++) free(matrix[i]);
    free(matrix);
    MPI_Request reduceReq;
    MPI_Status temp;
    int significantDiff = 1; // No difference if 0, i.e. convergence.
    double prevVal;
    while (significantDiff) { // Iterate until convergence.
        // Distribute local matrices -> Their share of rows + above and below rows
        significantDiff = 0; // Set convergence flag to 0.
        for (int i = 1; i < portion+1; i++) { // Starting at row 1 in local matrix.
            for (int j = 1; j < size - 1; j++) { // excl. vertical borders
                prevVal = uniqueMatrix[i][j];
                // Take average and round to specified precision.
                uniqueMatrix[i][j] = (uniqueMatrix[i-1][j] + uniqueMatrix[i+1][j]
                                      + uniqueMatrix[i][j-1] + uniqueMatrix[i][j+1]) / 4;
                // Expensive operation, only do as many times as necessary.
                if (!significantDiff && // Don't evaluate the rest if flag is set
                fabs(uniqueMatrix[i][j] - prevVal) >= precision) { // Change occurred
                    significantDiff = 1; // Set flag to true
                    // OR operation on convergence flag across all processors
                    MPI_Iallreduce(MPI_IN_PLACE, &significantDiff, 1, MPI_INT,
                                MPI_LOR, MPI_COMM_WORLD, &reduceReq);
                }
            }
        }
        // Only call for reduce again if flag is false, i.e. not called before
        if (!significantDiff) {
            MPI_Iallreduce(MPI_IN_PLACE, &significantDiff, 1, MPI_INT, MPI_LOR,
                        MPI_COMM_WORLD, &reduceReq);
        }
        MPI_Wait(&reduceReq, MPI_STATUS_IGNORE);

        if (significantDiff && concurrent) {
            if (!rank) {
                MPI_Sendrecv(uniqueMatrix[portion], size, MPI_DOUBLE, 1, 1,
                             uniqueMatrix[portion + 1], size, MPI_DOUBLE, 1, 1,
                             MPI_COMM_WORLD, &temp);
            }
            else if (rank == processorCount-1) {
                MPI_Sendrecv(uniqueMatrix[1], size, MPI_DOUBLE, rank-1, 1,
                             uniqueMatrix[0], size, MPI_DOUBLE, rank-1, 1,
                             MPI_COMM_WORLD, &temp);
            }
            else {
                MPI_Sendrecv(uniqueMatrix[1], size, MPI_DOUBLE, rank - 1, 1,
                             uniqueMatrix[portion + 1], size, MPI_DOUBLE,
                             rank + 1, 1,MPI_COMM_WORLD, &temp);
                MPI_Sendrecv(uniqueMatrix[portion], size, MPI_DOUBLE, rank + 1, 1,
                             uniqueMatrix[0], size, MPI_DOUBLE, rank - 1, 1,
                             MPI_COMM_WORLD, &temp);
            }
        }
    }
    int matrixSize = !rank ? size*size :
            (rank == processorCount-1 ? (portion + 1) * size : portion * size);
    double *flattenedMatrix = (double*) malloc((size_t) matrixSize * sizeof(double));
    if (flattenedMatrix == NULL) {
        MPI_Abort(MPI_COMM_WORLD, -1);
        return NULL;
    }
    start = 0;
    int end = portion + 1;
    if (rank) {
        start++; // Borrowed row at the top for Processor 1 onwards
    }
    if (rank == processorCount-1) end++; // Last processor deals with final row
    for (int i = start; i < end; i++) { // Only the computed portion
        for (int j = 0; j < size; j++) {
            flattenedMatrix[(i-start) * size + j] = uniqueMatrix[i][j];
        }
    }
    for (int i = 0; i < portion + 2; i++) free(uniqueMatrix[i]);
    free(uniqueMatrix);

    if (!rank) { // Only rank 0 manages the global communications
        // Compute counts and displacements for MPI_Gatherv.
        int* recvcounts = (int*) calloc(processorCount, sizeof(int));
        int gatherDispls[processorCount]; // Displacement for MPI_Gatherv
        int valuesWithRemainder = portion * size;
        // Rank 0 would always account for remainder
        int valuesWithoutRemainder = remainder ? (portion-1)*size : valuesWithRemainder;
        int currGatherDispls = 0; // Start row 0 col 0.

        /* Extra row for first and last processors */
        recvcounts[0] += size;
        recvcounts[processorCount-1] += size;
        if (concurrent) gatherDispls[1] += size; // Adjust displacement for rank 2.

        /* Addressing remainders -> additional counts and displs */
        for (int i = 0; i < remainder; i++) {
            gatherDispls[i] = currGatherDispls;
            recvcounts[i] += valuesWithRemainder;
            currGatherDispls += recvcounts[i];
        }

        for (int i = remainder; i < processorCount; i++) {
            gatherDispls[i] = currGatherDispls;
            recvcounts[i] += valuesWithoutRemainder;
            currGatherDispls += recvcounts[i];
        }
        // Coalesce partial matrices into a single one.
        MPI_Gatherv(MPI_IN_PLACE, matrixSize, MPI_DOUBLE, flattenedMatrix,
                    recvcounts, gatherDispls, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        free(recvcounts);
        /*
         * Correctness testing:
         * Test different values against the average of their neighbours
         * If the values match, then we can assume convergence has been reached
         * Hence, the program functions correctly.
         */
        // Value to check must not be part of the outer values and must be in range.
//        int valueToCheck = 231543;
//        printf("Value to test: %.3lf\tAbove: %.3lf\tBelow: %.3lf\tLeft: %.3lf\t"
//                   "Right: %.3lf\n", flattenedMatrix[valueToCheck],
//                   flattenedMatrix[valueToCheck-size], flattenedMatrix[valueToCheck+size],
//                   flattenedMatrix[valueToCheck-1], flattenedMatrix[valueToCheck+1]);
        return flattenedMatrix;
    }
    MPI_Gatherv(flattenedMatrix, matrixSize, MPI_DOUBLE, NULL,NULL, NULL,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(flattenedMatrix);
    return NULL;
}

int main (int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size = 2000; // Array size, can be changed.
    double outerVal = 1.0; // Outer value. Can be changed.
    double** matrix = (double**) malloc((size_t) size * sizeof(double*));
    if (matrix == NULL) {
        MPI_Finalize();
        return -1;
    }
    for (int i = 0; i < size; i++) {
        matrix[i] = (double*) calloc((size_t) size, sizeof(double));
        if (matrix[i] == NULL) {
            MPI_Finalize();
            return -1;
        }
        matrix[i][0] = outerVal;
        matrix[0][i] = outerVal;
    }

    double* resultMatrix;
    double start, end;
    start = MPI_Wtime();
    // Precision, array size, and boundary values can be altered.
    resultMatrix = perform_relaxation(matrix, size, 0.001);
    end = MPI_Wtime();
    if (resultMatrix == NULL) {
        MPI_Finalize();
        return 0;
    }
    printf("Time: %f\n", end - start);
    /* Write to file / print computed array - Testing */
//    FILE* file = fopen("results.out", "a");
//    for (int i = 0; i < size; i++) {
//        for (int j = 0; j < size; j++) {
//            printf("%.3lf\t", resultMatrix[i*size + j]);
//            fprintf(file, "%.3lf,", resultMatrix[(i*size)+j]);
//        }
//        printf("\n");
//    }
//    fprintf(file, "\n");
//    fclose(file);
    free(resultMatrix);
    MPI_Finalize();
    return 0;
}
