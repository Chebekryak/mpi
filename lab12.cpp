#include <mpi.h>
#include <stdio.h>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // задание 1, топология кольца
    
    MPI_Comm ring_comm;
    int dims[1] = {world_size}; // одномерная решетка
    int periods[1] = {1}; // периодическая топология (кольцо)
    int reorder = 1;
    
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, &ring_comm);
    
    int ring_rank, ring_size;
    MPI_Comm_rank(ring_comm, &ring_rank);
    MPI_Comm_size(ring_comm, &ring_size);

    // определяем соседей в кольце
    int left, right;
    MPI_Cart_shift(ring_comm, 0, 1, &left, &right);
    
    // процессы обмениваются данными
    int send_data = ring_rank;
    int recv_data;
    MPI_Sendrecv(&send_data, 1, MPI_INT, right, 0, &recv_data, 1, MPI_INT, left, 0, ring_comm, MPI_STATUS_IGNORE);

    printf("- кольцо: процесс %d отправил сообщение %d процессу %d и получил %d от процесса %d\n", ring_rank, send_data, right, recv_data, left);
    
    MPI_Comm_free(&ring_comm);
    

    // Синхронизация перед переходом ко второй части
    MPI_Barrier(MPI_COMM_WORLD);

    // задание 1, топология графа master-slave
    
    if (world_size >= 2) {
        MPI_Comm star_comm;
        int nnodes = world_size;
        vector<int> index(world_size + 1);
        vector<int> edges(world_size * 2);

        if (world_rank == 0) {
            index[0] = world_size - 1; // У мастера world_size - 1 соседей
            for (int i = 0; i < world_size; ++i) {
                edges[i] = i + 1; // Соседи мастера - все остальные процессы
                index[i + 1] = index[i] + 1;
            }
            for (int i = world_size; i < world_size * 2; ++i) {
                edges[i] = 0;
            }
        } else {
                index[0] = 0;
        } 
            
        
        // Рассылка информации о топологии
        MPI_Bcast(index.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(edges.data(), world_size - 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        int reorder = 1;
        MPI_Graph_create(MPI_COMM_WORLD, nnodes, index.data(), edges.data(), reorder, &star_comm);
        
        int star_rank;
        MPI_Comm_rank(star_comm, &star_rank);
        
        // обмен данными
        if (star_rank == 0) {
            // мастер получает данные от всех процессов
            for (int i = 1; i < world_size; i++) {
                int data;
                MPI_Recv(&data, 1, MPI_INT, i, 0, star_comm, MPI_STATUS_IGNORE);
                printf("- звезда: мастер получил %d от процесса %d\n", data, i);
            }
        } else {
            // слейв отправляют данные мастеру
            int data = star_rank;
            MPI_Send(&data, 1, MPI_INT, 0, 0, star_comm);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // проверим топологию
        int num_neighbors;
        MPI_Graph_neighbors_count(star_comm, world_rank, &num_neighbors);

        int neighbors[num_neighbors];
        MPI_Graph_neighbors(star_comm, world_rank, num_neighbors, neighbors);

        printf("Процесс %d имеет %d соседей: ", world_rank, num_neighbors);
        for (int i = 0; i < num_neighbors; i++) {
            printf("%d ", neighbors[i]);
        }
        printf("\n");
        MPI_Comm_free(&star_comm);
    }
    


    MPI_Finalize();
    return 0;
}