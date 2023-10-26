#include <iostream>
#include <vector>
#include <mpi.h>
#include <cmath>

using namespace std;
using i64 = int64_t;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int shift = 15;
    const int a = 1 << shift;
    const long long n = a * a;
    int reps = 100;
    i64 local_size = n / size;
    i64 start_idx = rank * local_size;
    i64 end_idx = start_idx + local_size;
	MPI_Barrier(MPI_COMM_WORLD);
	double ts0 = MPI_Wtime();
    std::vector<std::vector<double> > A_skinny(local_size, std::vector<double>(4));
    std::vector<std::vector<long long> > I_skinny(local_size, std::vector<long long>(4));
	std::vector<std::vector<std::vector<long long> > > send_res_mat(size, std::vector<std::vector<long long> >(size, std::vector<long long>()));
    std::vector<double> v_old(n);
    std::vector<double> v_new(n);

	// Populates I_skinny and send_res_mat
	for (i64 i = start_idx; i < end_idx; i++) {
		i64 idx[4];
		i64 adj_rank;

		idx[0] = (0 <= i - 1 && i % (a) != 0) ? (i - 1) : i;
		idx[1] = i;
		idx[2] = (i + 1 <= n && (i + 1) % (a) != 0) ? (i + 1) : i;
		idx[3] = (i % 2 == 0) ? (a + i + 1 <= n ? (a + i + 1) : i) : (0 <= i - a - 1 ? (i - a - 1) : i);

		I_skinny[i - start_idx][0] = idx[0];
		I_skinny[i - start_idx][1] = idx[1];
		I_skinny[i - start_idx][2] = idx[2];
		I_skinny[i - start_idx][3] = idx[3];

		for (i64 j = 0; j < 4; j++) {
			if (start_idx > idx[j]) { adj_rank = (rank - 1 + size) % size;} 
			else if (idx[j] > end_idx) { adj_rank = (rank + 1) % size;} 
			else { continue; }
			send_res_mat[rank][adj_rank].push_back(i);
			send_res_mat[adj_rank][rank].push_back(idx[j]);
		}
	}

    for (i64 i = 0; i < local_size; i++) {
		A_skinny[i][0] = 0.2;
		A_skinny[i][1] = 0.4;
		A_skinny[i][2] = 0.2;
		A_skinny[i][3] = 0.2;
    }

	i64 send_amount;
	
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

	std::vector<std::vector<double> > send_buffer(size);
	std::vector<std::vector<double> > recv_buffer(size);
	for (i64 dest = 0; dest < size; dest++) {
		if (dest == rank) { continue; }
		if (send_res_mat[rank][dest].size() == 0) { continue; }
		send_buffer[dest].resize(send_res_mat[rank][dest].size());
		recv_buffer[dest].resize(send_res_mat[rank][dest].size());
	}

	double t0, tcomm = 0.0, tcomp = 0.0;
	MPI_Barrier(MPI_COMM_WORLD);
	t0 = MPI_Wtime();

	for (i64 k = 0; k < reps; k++) {
		double tc1 = MPI_Wtime();
		for (i64 i = 0; i < local_size; i++) {
			v_new[i + start_idx] = 0.0;
			for (i64 j = 0; j < 4; j++) {
				v_new[i + start_idx] += A_skinny[i][j] * v_old[I_skinny[i][j]];
			}
		}

		send_amount = 0;
		for (i64 dest = 0; dest < size; dest++) {
			if (dest == rank) { continue; }
			if (send_res_mat[rank][dest].size() == 0) { continue; }
			for (i64 j = 0; j < send_res_mat[rank][dest].size(); j++) {
				send_buffer[dest][j] = (v_new[send_res_mat[rank][dest][j]]);
			}
			send_amount++;
		}
		double tc2 = MPI_Wtime();


		MPI_Request send_requests[send_amount];
		MPI_Request recv_requests[send_amount];

		send_amount = 0;
		for (i64 dest = 0; dest < size; dest++) {
			if (dest == rank) { continue; }
			if (send_res_mat[rank][dest].size() == 0) { continue; }
			MPI_Isend(send_buffer[dest].data(), send_buffer[dest].size(), MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &send_requests[send_amount]);
			MPI_Irecv(recv_buffer[dest].data(), recv_buffer[dest].size(), MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &recv_requests[send_amount]);
			send_amount++;
		}

		double tc3 = MPI_Wtime();
		tcomm += tc3 - tc2;
		tcomp += tc2 - tc1;

		MPI_Waitall(send_amount, send_requests, MPI_STATUSES_IGNORE);
    	MPI_Waitall(send_amount, recv_requests, MPI_STATUSES_IGNORE);
		
		for (i64 dest = 0; dest < size; dest++) {
			if (dest == rank) { continue; }
			if (send_res_mat[rank][dest].size() == 0) { continue; }
			for (i64 j = 0; j < send_res_mat[rank][dest].size(); j++) {
				v_new[send_res_mat[dest][rank][j]] = recv_buffer[dest][j];
			}
		}

		std::swap(v_new, v_old);
	}

	MPI_Allgather(v_old.data() + start_idx, local_size, MPI_DOUBLE, v_old.data(), local_size, MPI_DOUBLE, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	double t1 = MPI_Wtime();

	double l2 = 0.0;
	for (i64 j = 0; j < n; j++)
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