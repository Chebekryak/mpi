#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <vector>

using namespace std;

int Rand() {
    vector<int> list{ -1, 0, 1, 2, 3, 4, 5};
    int i = rand() % 7;
    return list[i];
}

int main(int argc, char** argv) {
    srand(time(NULL));

    MPI_Init(&argc, &argv);
    MPI_Status status;

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   
    int counter = 0;

    bool flag = true;

    while (flag) {
        if (rank == 0) {
            int received_value;
            MPI_Recv(&received_value, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (received_value == -1) {
                printf("Процесс 0 получил -1 от процесса %d\t Текущее значение счетчика: %d\n", status.MPI_SOURCE, counter);
                flag = false;
            }
            else {
                counter++;
                printf("Процесс 0 получил %d от процесса %d\tСчетчик увеличен до: %d\n", received_value, status.MPI_SOURCE, counter);
            }
            for (int i = 1; i < size; i++) MPI_Send(&flag, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        else {
            int value = Rand();
            MPI_Send(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            printf("Процесс %d отправил значение %d\n", rank, value)
            MPI_Recv(&flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        }
    }

    MPI_Finalize();
    return 0;
}