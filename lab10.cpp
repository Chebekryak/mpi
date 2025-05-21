#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define MAX_DIGITS 1000

typedef struct {
    int length;
    char digits[MAX_DIGITS];
} LongInt;

void multiply(LongInt* a, LongInt* b, LongInt* result) {
    memset(result->digits, '0', MAX_DIGITS);
    result->length = a->length + b->length;
    result->digits[result->length] = '\0';

    for (int i = a->length - 1; i >= 0; i--) {
        int carry = 0;
        for (int j = b->length - 1; j >= 0; j--) {
            int pos = i + j + 1;
            int prod = (a->digits[i] - '0') * (b->digits[j] - '0') + (result->digits[pos] - '0') + carry;
            carry = prod / 10;
            result->digits[pos] = (prod % 10) + '0';
        }
        result->digits[i] += carry;
    }

    // удаляем ведущие нули
    int start = 0;
    while (start < result->length - 1 && result->digits[start] == '0') {
        start++;
    }
    if (start > 0) {
        memmove(result->digits, result->digits + start, result->length - start + 1);
        result->length -= start;
    }
}

// функция для редукции
void longint_multiply(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype) {
    LongInt* a = (LongInt*)invec;
    LongInt* b = (LongInt*)inoutvec;
    LongInt temp;
    
    multiply(a, b, &temp);
    *b = temp;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // инициализируем наш тип данных
    MPI_Datatype longint_type;
    MPI_Type_contiguous(sizeof(LongInt), MPI_BYTE, &longint_type);
    MPI_Type_commit(&longint_type);

    // инициализируем функцию для редукции
    MPI_Op multiply_op;
    MPI_Op_create(longint_multiply, 1, &multiply_op);

    int n = 0;
    LongInt* numbers = NULL;
    int* counts = NULL;
    int* displs = NULL;

    if (rank == 0) {
        printf("Введите количество чисел: ");
        fflush(stdout);
        scanf("%d", &n);

        numbers = (LongInt*)malloc(n * sizeof(LongInt));
        for (int i = 0; i < n; i++) {
            printf("Введите число %d: ", i+1);
            fflush(stdout);
            scanf("%s", numbers[i].digits);
            numbers[i].length = strlen(numbers[i].digits);
        }

        counts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        
        int base = n / size;
        int rem = n % size;
        for (int i = 0; i < size; i++) {
            counts[i] = (i < rem) ? base + 1 : base;
            displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
        }
    }

    // рассылаем количество чисел
    int local_count;
    MPI_Scatter(counts, 1, MPI_INT, &local_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // получаем локальные числа
    LongInt* local_nums = (LongInt*)malloc(local_count * sizeof(LongInt));
    MPI_Scatterv(numbers, counts, displs, longint_type, local_nums, local_count, longint_type, 0, MPI_COMM_WORLD);

    // локальный результат умножений
    LongInt local_result;
    if (local_count > 0) {
        local_result = local_nums[0];
        for (int i = 1; i < local_count; i++) {
            LongInt temp;
            multiply(&local_result, &local_nums[i], &temp);
            local_result = temp;
        }
    } else {
        local_result.length = 1;
        strcpy(local_result.digits, "1");
    }

    // глобальное умножение
    LongInt global_result;
    MPI_Reduce(&local_result, &global_result, 1, longint_type, multiply_op, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Результат умножения: %s\n", global_result.digits);
        free(numbers);
        free(counts);
        free(displs);
    }

    free(local_nums);
    MPI_Op_free(&multiply_op);
    MPI_Type_free(&longint_type);
    MPI_Finalize();
    return 0;
}