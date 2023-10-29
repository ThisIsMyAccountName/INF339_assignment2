#include <iostream>
#include <vector>
#include <mpi.h>
#include <map>
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
    std::vector<double> v_old(local_size, 0.0);
    std::vector<double> v_new(local_size, 0.0);

	// Populates I_skinny and send_res_mat
    for (int i = start_idx; i < end_idx; i++) {
		int idx0 = 0 <= i - 1 && i % (a) != 0 ? (i - 1) : i;
		int idx1 = i;
		int idx2 = i + 1 <= n && (i + 1) % (a) != 0 ? (i + 1) : i;
		int idx3;
		if (i % 2 == 0){ idx3 = a + i + 1 <= n ? (a + i + 1) : i; }
		else { idx3 = 0 <= i - a - 1 ? (i - a - 1) : i; }
        I_skinny[i - start_idx][0] = idx0 % local_size;
		I_skinny[i - start_idx][1] = idx1 % local_size;
		I_skinny[i - start_idx][2] = idx2 % local_size;
		// Dont mod here, need value later
		I_skinny[i - start_idx][3] = idx3;
		if (start_idx > idx0) { 
			send_res_mat[rank][(rank - 1) % size].push_back(i);
			send_res_mat[(rank - 1) % size][rank].push_back(idx0);
			} 
		else if (idx0 > end_idx) {
			send_res_mat[rank][(rank + 1) % size].push_back(i);
			send_res_mat[(rank + 1) % size][rank].push_back(idx0);
			}
		if (start_idx > idx1) {
			send_res_mat[rank][(rank - 1) % size].push_back(i);
			send_res_mat[(rank - 1) % size][rank].push_back(idx1);
			}
		else if (idx1 > end_idx) {
			send_res_mat[rank][(rank + 1) % size].push_back(i);
			send_res_mat[(rank + 1) % size][rank].push_back(idx1);
			}
		if (start_idx > idx2) {
			send_res_mat[rank][(rank - 1) % size].push_back(i);
			send_res_mat[(rank - 1) % size][rank].push_back(idx2);
			}
		else if (idx2 > end_idx) {
			send_res_mat[rank][(rank + 1) % size].push_back(i);
			send_res_mat[(rank + 1) % size][rank].push_back(idx2);
			}
		if (start_idx > idx3) {
			send_res_mat[rank][(rank - 1) % size].push_back(i);
			send_res_mat[(rank - 1) % size][rank].push_back(idx3);
			}
		else if (idx3 > end_idx) {
			send_res_mat[rank][(rank + 1) % size].push_back(i);
			send_res_mat[(rank + 1) % size][rank].push_back(idx3);
			}	
	
    }
	// Populates A_skinny
    for (int i = 0; i < local_size; i++) {
		A_skinny[i][0] = 0.1;
		A_skinny[i][1] = 0.7;
		A_skinny[i][2] = 0.1;
		A_skinny[i][3] = 0.1;
    }
    
	std::vector<std::vector<int> > rank_send(size, std::vector<int>());
	std::vector<std::vector<int> > rank_recv(size, std::vector<int>());
	std::map<int, int> rank_map;
	int next_idx = 0;
	int prev_idx = local_size;
	for (int i = 0; i < send_res_mat[rank].size(); i++) {
		v_new.push_back(0.0);
		v_old.push_back(0.0);
		if (i == rank) { continue; }
		if (send_res_mat[rank][i].size() == 0) { continue; }
		for (int j = 0; j < send_res_mat[i][rank].size(); j++) {
			rank_send[i].push_back(send_res_mat[rank][i][j] % local_size);
			rank_recv[i].push_back(prev_idx);
			rank_map[send_res_mat[i][rank][j]] = prev_idx;
			prev_idx++;
		}
	}

	for (int i = 0; i < I_skinny.size(); i++) {
		if (I_skinny[i][3] >= start_idx && I_skinny[i][3] < end_idx){
			I_skinny[i][3] %= local_size;
		} 
		else {
			I_skinny[i][3] = rank_map[I_skinny[i][3]];
		}
	}

	if (rank == 0){
		v_old[0] = 1;
	}
	MPI_Barrier(MPI_COMM_WORLD);
	double ts1 = MPI_Wtime();
	double init_time = ts1 - ts0;
	if (rank == 0){
		cout << init_time << endl;
	}

	std::vector<std::vector<double> > send_buffer(size);
	std::vector<std::vector<double> > recv_buffer(size);
	for (int dest = 0; dest < size; dest++) {
		if (dest == rank) { continue; }
		if (rank_send[dest].size() == 0) { continue; }
		send_buffer[dest].resize(rank_send[dest].size());
		recv_buffer[dest].resize(rank_recv[dest].size());
	}

	double t0, tcomm = 0.0, tcomp = 0.0;
	MPI_Barrier(MPI_COMM_WORLD);
	t0 = MPI_Wtime();

	for (int k = 0; k < reps; k++) {
		double tc1 = MPI_Wtime();
		for (int i = 0; i < local_size; i++) {
			v_new[i] = 0.0;
			for (int j = 0; j < 4; j++) {
				v_new[i] += A_skinny[i][j] * v_old[I_skinny[i][j]];
			}
		}


		int send_amount = 0;
		for (int dest = 0; dest < size; dest++) {
			if (dest == rank) { continue; }
			if (rank_send[dest].size() == 0) { continue; }
			for (int j = 0; j < rank_send[dest].size(); j++) {
				send_buffer[dest][j] = v_new[rank_send[dest][j]];
			}
			send_amount++;
		}

		MPI_Request send_requests[send_amount];
		MPI_Request recv_requests[send_amount];

		double tc2 = MPI_Wtime();

		send_amount = 0;
		for (int dest = 0; dest < size; dest++) {
			if (dest == rank) { continue; }
			if (rank_send[dest].size() == 0) { continue; }
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
			if (rank_send[dest].size() == 0) { continue; }
			for (int j = 0; j < send_res_mat[rank][dest].size(); j++) {
				v_new[rank_recv[dest][j]] = recv_buffer[dest][j];
			}
		}
		double tc4 = MPI_Wtime();
		tcomp += tc4 - tc3;
		std::swap(v_new, v_old);
	}
	
	// std::vector<double> v_fin(n);
	// if (rank == 0) {
	// 	for (int i = 0; i < local_size; i++) {
	// 		v_fin[i] = v_old[i];
	// 	}
	// 	for (int i = 1; i < size; i++) {
	// 		MPI_Recv(v_fin.data() + i * local_size, local_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	// 	}
	// }
	// else {
	// 	MPI_Send(v_old.data(), local_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	// }

	MPI_Barrier(MPI_COMM_WORLD);
	double t1 = MPI_Wtime();

	double l2 = 0.0;
	for (int j = 0; j < local_size; j++)
		l2 += v_old[j] * v_old[j];

	l2 = sqrt(l2);

	double ops = (long long)n * 8ll * 100ll; // 4 multiplications and 4 additions
	double time = t1 - t0;
	
	int send_count = 0;
	for (int j = 0; j < rank_send.size(); j++)
		send_count += rank_send[j].size();

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