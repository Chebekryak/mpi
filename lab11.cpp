#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cstdio>

using namespace std;

double determinant(vector<vector<double>>& matrix, int n) {
    double det = 1.0;
    for (int i = 0; i < n; ++i) {
        if (matrix[i][i] == 0) {
            for (int j = i + 1; j < n; ++j) {
                if (matrix[j][i] != 0) {
                    swap(matrix[i], matrix[j]);
                    det *= -1;
                    break;
                }
            }
            if (matrix[i][i] == 0) return 0;
        }
        det *= matrix[i][i];
        for (int j = i + 1; j < n; ++j) {
            double factor = matrix[j][i] / matrix[i][i];
            for (int k = i; k < n; ++k) {
                matrix[j][k] -= factor * matrix[i][k];
            }
        }
    }
    return det;
}

void printMatrix(const vector<vector<double>>& matrix, const vector<double>& b) {
    int n = matrix.size();
    printf("Матрица системы:\n");
    for (int i = 0; i < n; ++i) {
        printf("| ");
        for (int j = 0; j < n; ++j) {
            printf("%6.2f ", matrix[i][j]);
        }
        printf(" | = | %6.2f |\n", b[i]);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // создаем группы и коммуникаторы
    MPI_Group world_group, worker_group, master_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    
    int exclude_ranks[1] = {0};
    MPI_Group_excl(world_group, 1, exclude_ranks, &worker_group); // создаем группу для рабочих процессов (все кроме 0)
    
    int include_ranks[1] = {0};
    MPI_Group_incl(world_group, 1, include_ranks, &master_group); // создаем группу только для master-процесса (ранг 0)
    
    MPI_Comm worker_comm, master_comm; // создаем коммуникаторы для этих групп
    MPI_Comm_create(MPI_COMM_WORLD, worker_group, &worker_comm);
    MPI_Comm_create(MPI_COMM_WORLD, master_group, &master_comm);

    int A;
    vector<vector<double>> matrix;
    vector<double> b;

    if (master_comm != MPI_COMM_NULL) {
        printf("Введите размер матрицы A: ");
        fflush(stdout);
        scanf("%d", &A);

        srand(time(0));
        matrix.resize(A, vector<double>(A));
        b.resize(A);

        for (int i = 0; i < A; ++i) {
            for (int j = 0; j < A; ++j) {
                matrix[i][j] = rand() % 10 + 1;
            }
        }
        for (int i = 0; i < A; ++i) {
            b[i] = rand() % 10 + 1;
        }

        printMatrix(matrix, b);
    }

    // передаем размер матрицы
    MPI_Bcast(&A, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (worker_comm != MPI_COMM_NULL) {
        matrix.resize(A, vector<double>(A));
        b.resize(A);
    }

    // передаем матрицу и вектор b
    for (int i = 0; i < A; ++i) {
        MPI_Bcast(matrix[i].data(), A, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(b.data(), A, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // вычисляем определитель
    double detMain = 0;
    if (master_comm != MPI_COMM_NULL) {
        vector<vector<double>> tempMatrix = matrix;
        detMain = determinant(tempMatrix, A);
        
        printf("\nГлавный определитель (det A) = %.2f\n", detMain);
        
        if (fabs(detMain) < 1e-9) {
            fprintf(stderr, "Определитель равен нулю, система не имеет решения\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    // передаем определитель
    MPI_Bcast(&detMain, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // распределяем вычисления
    vector<double> detSub(A, 0.0);

    if (worker_comm != MPI_COMM_NULL) {
        int worker_rank, worker_size;
        MPI_Comm_rank(worker_comm, &worker_rank);
        MPI_Comm_size(worker_comm, &worker_size);

        vector<pair<int, double>> computed_values;

        for (int var = worker_rank; var < A; var += worker_size) {
            vector<vector<double>> subMatrix = matrix;
            for (int i = 0; i < A; ++i) subMatrix[i][var] = b[i];
            computed_values.emplace_back(var, determinant(subMatrix, A));
        }

        int count = computed_values.size();
        MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        
        MPI_Send(computed_values.data(), count * 2, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    if (master_comm != MPI_COMM_NULL) {
        for (int src = 1; src < world_size; src++) {
            int count;
            MPI_Recv(&count, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            vector<pair<int, double>> received(count);
            MPI_Recv(received.data(), count * 2, MPI_DOUBLE, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (auto& [idx, val] : received) {
                detSub[idx] = val;
            }
        }

        vector<double> x(A);
        printf("\nРешение системы методом Крамера:\n");
        for (int i = 0; i < A; ++i) {
            x[i] = detSub[i] / detMain;
            printf("x[%d] = det A%d / det A = %.2f / %.2f = %.2f\n", 
                i, i, detSub[i], detMain, x[i]);
        }
    }

    if (worker_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&worker_comm);
    }
    if (master_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&master_comm);
    }
    MPI_Group_free(&worker_group);
    MPI_Group_free(&master_group);
    MPI_Group_free(&world_group);

    MPI_Finalize();
    return 0;
}