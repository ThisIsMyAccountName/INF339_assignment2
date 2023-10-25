#include <iostream>
#include <vector>
#include <mpi.h>

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int shift = 10;
    const int a = 1 << shift;
    const int n = a * a;
    int reps = 1000;
    int local_size = n / size;
    int start_idx = rank * local_size;
    int end_idx = start_idx + local_size;
	int sep_size = 0;
    std::vector<std::vector<double> > A_skinny(local_size, std::vector<double>(4));
    std::vector<std::vector<int> > I_skinny(local_size, std::vector<int>(4));
    std::vector<double> v_old(n);
    std::vector<double> v_new(n);
    for (int i = start_idx; i < end_idx; i++) {
		int idx0 = 0 <= i - 1 && i % (a) != 0 ? (i - 1) : i;
		int idx1 = i;
		int idx2 = i + 1 <= n && (i + 1) % (a) != 0 ? (i + 1) : i;
		int idx3;
		if (i % 2 == 0){ idx3 = a + i + 1 <= n ? (a + i + 1) : i; }
		else { idx3 = 0 <= i - a - 1 ? (i - a - 1) : i; }
        I_skinny[i - start_idx][0] = idx0;
		I_skinny[i - start_idx][1] = idx1;
		I_skinny[i - start_idx][2] = idx2;
		I_skinny[i - start_idx][3] = idx3;
    }
    
    for (int i = 0; i < local_size; i++) {
        A_skinny[i][0] = 0.2;
        A_skinny[i][1] = 0.4;
        A_skinny[i][2] = 0.2;
        A_skinny[i][3] = 0.2;
    }
    if (rank == 0) {
        v_old[0] = 1;
    }

    MPI_Bcast(v_old.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double time_accumalator = 0;
    double start_time, end_time;
    MPI_Barrier(MPI_COMM_WORLD);

    for (int k = 0; k < reps; k++) {
        for (int i = 0; i < local_size; i++) {
            v_new[i + start_idx] = 0.0;
            for (int j = 0; j < 4; j++) {
                v_new[i + start_idx] += A_skinny[i][j] * v_old[I_skinny[i][j]];
            }
        }
        start_time = MPI_Wtime();
        MPI_Allgather(v_new.data() + start_idx, local_size, MPI_DOUBLE, v_new.data(), local_size, MPI_DOUBLE, MPI_COMM_WORLD);
        end_time = MPI_Wtime();
        time_accumalator += end_time - start_time;
        std::swap(v_new, v_old);
	
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        cout << time_accumalator << endl;
    }

    MPI_Finalize();

    return 0;
}
