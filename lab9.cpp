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
    MPI_Init(&argc, &argv);
    MPI_Status status;

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    srand(time(NULL) + rank);
   
    int counter = 0;
    vector<int> messages(size);
    bool flag = true;

    while (flag) {

        int value = Rand();
        MPI_Gather(&value, 1, MPI_INT, messages.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            for (int i = 1; i < size; i++) {
                int received_value = messages[i];
                if (received_value == -1) {
                    printf("Процесс 0 получил -1 от процесса %d\t Текущее значение счетчика: %d\n", i, counter);
                    flag = false;
                }
                else {
                    counter++;
                    printf("Процесс 0 получил %d от процесса %d\tСчетчик увеличен до: %d\n", received_value, i, counter);
                }
            }
        }

        MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}