#include <iostream>
#include <fstream>
#include <mpi.h>

double function(double x) {
    return 4.0 / (1.0 + x * x);
}

double compute_elementary_split(double delta_x, double left_border, double right_border) {    
    return (function(left_border) + function(right_border)) / 2.0 * delta_x;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    int N = atoi(argv[1]);
    double delta_x = 1.0 / N;

    std::ofstream out("out.txt");

    MPI_Status status;
    if (world_rank == 0) {
        // Последовательное вычисление интеграла
        double sequence_start_time = MPI_Wtime();

        double sequence_value = 0.0;

        for (int i = 0; i < N; ++i) {
            sequence_value += compute_elementary_split(delta_x, i * delta_x, (i + 1) * delta_x);
        }

        double sequence_end_time = MPI_Wtime();

        out << sequence_end_time - sequence_start_time << '\n';

        // Параллельное вычисление интеграла
        double parallel_start_time = MPI_Wtime();

        int busy_processes_count = N % world_size;
        int busy_processes_parts_count = N / world_size + 1;
        int free_processes_count = world_size - busy_processes_count;
        int free_processes_parts_count = N / world_size;

        for (int i = 0; i < busy_processes_count; ++i) {
            int left_index = i * busy_processes_parts_count;
            MPI_Send(&left_index, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
            int right_index = (i + 1) * busy_processes_parts_count;
            MPI_Send(&right_index, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
        }

        for (int i = 1; i < free_processes_count; ++i) {
            int left_index = busy_processes_count * busy_processes_parts_count + i * free_processes_parts_count;
            MPI_Send(&left_index, 1, MPI_INT, busy_processes_count + i, 0, MPI_COMM_WORLD);
            int right_index = busy_processes_count * busy_processes_parts_count + (i + 1) * free_processes_parts_count;
            MPI_Send(&right_index, 1, MPI_INT, busy_processes_count + i, 0, MPI_COMM_WORLD);
        }

        double parallel_value = 0.0;
        int left_index = busy_processes_count * busy_processes_parts_count;
        int right_index = busy_processes_count * busy_processes_parts_count + free_processes_parts_count;

        for (int i = left_index; i < right_index; ++i) {
            parallel_value += compute_elementary_split(delta_x, i * delta_x, (i + 1) * delta_x);
        }

        for (int i = 1; i < world_size; ++i) {
            double process_value;
            MPI_Recv(&process_value, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            parallel_value += process_value;
        }

        double parallel_end_time = MPI_Wtime();

        out << parallel_end_time - parallel_start_time << '\n';
        out << N << '\n' << world_size << '\n';

        std::cout << parallel_value << '\n';
        std::cout << sequence_value << '\n';
    } else {
        int left_index;
        MPI_Recv(&left_index, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        int right_index;
        MPI_Recv(&right_index, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        double value = 0.0;

        for (int i = left_index; i < right_index; ++i) {
            value += compute_elementary_split(delta_x, i * delta_x, (i + 1) * delta_x);
        }

        std::cout << world_rank << ' ' << value << '\n';
        MPI_Send(&value, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}