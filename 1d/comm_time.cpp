#include <iostream>
#include <vector>
#include <mpi.h>
#include <numeric>

using namespace std;


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int shift = 3;
    const int a = 1 << shift;
    const int n = a * a;
    int reps = atoi(argv[1]);
    int local_size = n / size;
    int start_idx = rank * local_size;
    int end_idx = start_idx + local_size;
	int sep_size = 0;
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

	// Populates A_skinny
    for (int i = 0; i < local_size; i++) {
		if (i % 2 == 0){
			A_skinny[i][0] = 0.2;
			A_skinny[i][1] = 0.4;
			A_skinny[i][2] = 0.2;
			A_skinny[i][3] = 0.2;
		}
		else {
			A_skinny[i][0] = 0.2;
			A_skinny[i][1] = 0.4;
			A_skinny[i][2] = 0.2;
			A_skinny[i][3] = 0.2;
		}
    }
    

    if (rank == 0) {
        v_old[0] = 1;
    }

    MPI_Bcast(v_old.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << 1 << endl;
    }


	std::vector<std::vector<double> > send_buffer(size);
	std::vector<std::vector<double> > recv_buffer(size);
	for (int dest = 0; dest < size; dest++) {
		if (dest == rank) { continue; }
		send_buffer[dest].resize(send_res_mat[rank][dest].size());
		recv_buffer[dest].resize(send_res_mat[rank][dest].size());
	}

	int send_amount;
	double time_accumulator = 0;
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
		send_amount = 0;
		for (int dest = 0; dest < size; dest++) {
			if (dest == rank) { continue; }
			if (send_res_mat[rank][dest].size() == 0) { continue; }
			for (int j = 0; j < send_res_mat[rank][dest].size(); j++) {
				send_buffer[dest][j] = (v_new[send_res_mat[rank][dest][j]]);
			}
			send_amount++;
		}
	
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

		for (int dest = 0; dest < size; dest++) {
			if (dest == rank) { continue; }
			for (int j = 0; j < send_res_mat[rank][dest].size(); j++) {
				v_new[send_res_mat[dest][rank][j]] = recv_buffer[dest][j];
			}
		}
		end_time = MPI_Wtime();
		time_accumulator += end_time - start_time;
		

		std::swap(v_new, v_old);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	end_time = MPI_Wtime();
	if (rank == 0) {
		cout << time_accumulator << endl;
	}
	
	MPI_Finalize();
	return 0;
}