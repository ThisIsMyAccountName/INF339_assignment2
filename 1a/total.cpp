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
    int reps = 100;
    int local_size = n / size;
    int start_idx = rank * local_size;
    int end_idx = start_idx + local_size;
    MPI_Barrier(MPI_COMM_WORLD);
	double ts0 = MPI_Wtime();

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
    MPI_Barrier(MPI_COMM_WORLD);
	double ts1 = MPI_Wtime();
	double init_time = ts1 - ts0;
	if (rank == 0){
		cout << init_time << endl;
	}
    MPI_Bcast(v_old.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double t0, tcomm = 0.0, tcomp = 0.0;
	MPI_Barrier(MPI_COMM_WORLD);
	t0 = MPI_Wtime();

    for (int k = 0; k < reps; k++) {
        double tc1 = MPI_Wtime();
        for (int i = 0; i < local_size; i++) {
            v_new[i + start_idx] = 0.0;
            for (int j = 0; j < 4; j++) {
                v_new[i + start_idx] += A_skinny[i][j] * v_old[I_skinny[i][j]];
            }
        }
        double tc2 = MPI_Wtime();
        MPI_Allgather(v_new.data() + start_idx, local_size, MPI_DOUBLE, v_new.data(), local_size, MPI_DOUBLE, MPI_COMM_WORLD);
        double tc3 = MPI_Wtime();
		tcomm += tc3 - tc2;
		tcomp += tc2 - tc1;

        std::swap(v_new, v_old);
	
    }
    MPI_Barrier(MPI_COMM_WORLD);
	double t1 = MPI_Wtime();

	double l2 = 0.0;
	for (int j = 0; j < n; j++)
		l2 += v_old[j] * v_old[j];

	l2 = sqrt(l2);

	double ops = (long long)n * 8ll * 100ll; // 4 multiplications and 4 additions
	double time = t1 - t0;
	
	if (rank == 0)
        {
            printf("%lfs (%lfs, %lfs), %lf GFLOPS, %lf GBs mem, %lf GBs comm, L2 = %lf\n",
                   time, tcomp, tcomm, (ops / time) / 1e9, (n * 64.0 * 100.0 / tcomp) / 1e9, ((local_size * (size - 1)) * 8.0 * size * 100.0 / tcomm) / 1e9, l2);
        }

    MPI_Finalize();

    return 0;
}
