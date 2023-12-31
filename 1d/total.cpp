#include <iostream>
#include <vector>
#include <mpi.h>
#include <cmath>
using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int shift = 15;
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
	std::vector<std::vector<std::vector<int> > > send_res_mat(size, std::vector<std::vector<int> >(size, std::vector<int>()));
    std::vector<double> v_old(n);
    std::vector<double> v_new(n);

	// Populates I_skinny and send_res_mat
	for (int i = start_idx; i < end_idx; i++) {
		int idx[4];
		int adj_rank;

		idx[0] = (0 <= i - 1 && i % (a) != 0) ? (i - 1) : i;
		idx[1] = i;
		idx[2] = (i + 1 <= n && (i + 1) % (a) != 0) ? (i + 1) : i;
		idx[3] = (i % 2 == 0) ? (a + i + 1 <= n ? (a + i + 1) : i) : (0 <= i - a - 1 ? (i - a - 1) : i);

		I_skinny[i - start_idx][0] = idx[0];
		I_skinny[i - start_idx][1] = idx[1];
		I_skinny[i - start_idx][2] = idx[2];
		I_skinny[i - start_idx][3] = idx[3];

		for (int j = 0; j < 4; j++) {
			if (start_idx > idx[j]) { adj_rank = (rank - 1 + size) % size;} 
			else if (idx[j] > end_idx) { adj_rank = (rank + 1) % size;} 
			else { continue; }
			send_res_mat[rank][adj_rank].push_back(i);
			send_res_mat[adj_rank][rank].push_back(idx[j]);
		}
	}

    for (int i = 0; i < local_size; i++) {
		A_skinny[i][0] = 0.1;
		A_skinny[i][1] = 0.7;
		A_skinny[i][2] = 0.1;
		A_skinny[i][3] = 0.1;
	}
	
	int send_amount;
	
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
	for (int dest = 0; dest < size; dest++) {
		if (dest == rank) { continue; }
		if (send_res_mat[rank][dest].size() == 0) { continue; }
		send_buffer[dest].resize(send_res_mat[rank][dest].size());
		recv_buffer[dest].resize(send_res_mat[rank][dest].size());
	}

	double t0, tcomm = 0.0, tcomp = 0.0;
	MPI_Barrier(MPI_COMM_WORLD);
	t0 = MPI_Wtime();

	for (int k = 0; k < reps; k++) {
		double tc1 = MPI_Wtime();
		for (int i = 0; i < local_size; i++) {
			v_new[i + start_idx] = A_skinny[i][0] * v_old[I_skinny[i][0]] 
					+ A_skinny[i][1] * v_old[I_skinny[i][1]]
					+ A_skinny[i][2] * v_old[I_skinny[i][2]]
					+ A_skinny[i][3] * v_old[I_skinny[i][3]];
			}

		send_amount = 0;
		for (int dest = 0; dest < size; dest++) {
			if (dest == rank) { continue; }
			if (send_res_mat[rank][dest].size() == 0) { continue; }
			for (int j = 0; j < send_res_mat[rank][dest].size(); j++) {
				send_buffer[dest][j] = (v_new[send_res_mat[rank][dest][j]]);
			}
			send_amount++;
		}
		double tc2 = MPI_Wtime();


		MPI_Request send_requests[send_amount];
		MPI_Request recv_requests[send_amount];

		send_amount = 0;
		for (int dest = 0; dest < size; dest++) {
			if (dest == rank) { continue; }
			if (send_res_mat[rank][dest].size() == 0) { continue; }
			MPI_Isend(send_buffer[dest].data(), send_buffer[dest].size(), MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &send_requests[send_amount]);
			MPI_Irecv(recv_buffer[dest].data(), recv_buffer[dest].size(), MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &recv_requests[send_amount]);
			send_amount++;
		}


		MPI_Waitall(send_amount, send_requests, MPI_STATUSES_IGNORE);
    	MPI_Waitall(send_amount, recv_requests, MPI_STATUSES_IGNORE);

		double tc3 = MPI_Wtime();
		tcomm += tc3 - tc2;
		tcomp += tc2 - tc1;
		
		for (int dest = 0; dest < size; dest++) {
			if (dest == rank) { continue; }
			if (send_res_mat[rank][dest].size() == 0) { continue; }
			for (int j = 0; j < send_res_mat[rank][dest].size(); j++) {
				v_new[send_res_mat[dest][rank][j]] = recv_buffer[dest][j];
			}
		}
		double tc4 = MPI_Wtime();
		tcomp += tc4 - tc3;

		std::swap(v_new, v_old);
	}

	MPI_Allgather(v_old.data() + start_idx, local_size, MPI_DOUBLE, v_old.data(), local_size, MPI_DOUBLE, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	double t1 = MPI_Wtime();

	double l2 = 0.0;
	for (int j = 0; j < n; j++)
		l2 += v_old[j] * v_old[j];

	l2 = sqrt(l2);

	double ops = (long long)n * 8ll * 100ll; // 4 multiplications and 4 additions
	double time = t1 - t0;

	int send_count = 0;
	for (int j = 0; j < send_res_mat[rank].size(); j++)
		send_count += send_res_mat[rank][j].size();

	if (rank == 0)
	{
		std:vector<int> send_counts(size);
		MPI_Gather(&send_count, 1, MPI_INT, send_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
		int total_comm = 0;
		for (int j = 0; j < size; j++)
			total_comm += send_counts[j];
			
		printf("%lfs (%lfs, %lfs), %lf GFLOPS, %lf GBs mem, %lf GBs comm, L2 = %lf\n",
			time, tcomp, tcomm,
			(ops / time) / 1e9,
			(n * 64.0 * 100.0 / tcomp) / 1e9,
			(total_comm * 8.0 * 100.0 / tcomm) / 1e9,
			l2);        
	}
	else
	{
		MPI_Gather(&send_count, 1, MPI_INT, NULL, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}