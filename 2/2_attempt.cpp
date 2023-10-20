#include <iostream>
#include <vector>
#include <mpi.h>
#include <map>

using namespace std;

void print_skinny_matrix(
	const std::vector<std::vector<int> >& skinny,
	const int& n) {
	for (int i = 0; i < n; i++) {
		cout << skinny[i][0] << " " << skinny[i][1] << " " << skinny[i][2] << " " << skinny[i][3] << endl;
	}
}

void print_matrix(
    const std::vector<double>& v,
    const int& n,
    const int& a) {
    for (int i = 0; i < n; i++) {
        cout << v[i] << " ";
        if (i % a == a - 1) {
            cout << endl;
        }
    }
}

void print_vector_d(
	const std::vector<double>& v,
	const int& n) {
	for (int i = 0; i < n; i++) {
		cout << v[i] << " ";
	}
}
void print_vector_i(
	const std::vector<int>& v,
	const int& n) {
	for (int i = 0; i < n; i++) {
		cout << v[i] << " ";
	}
}

void print_3d_matrix(
	const std::vector<std::vector<std::vector<int> > >& matrix) {
    for (size_t i = 0; i < matrix.size(); i++) {
		cout << "rank: " << i << endl;
        for (size_t j = 0; j < matrix[i].size(); j++) {
            for (size_t k = 0; k < matrix[i][j].size(); k++) {
				// print the list where i is from j is to and k in values
				cout << "from: " << i << " to: " << j << " value: " << matrix[i][j][k] << endl;
            }
        }
    }
}

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
	// MPI_Barrier(MPI_COMM_WORLD);
	// if (rank == 0) {
	// 	cout << "I_skinny_reordered, rank: " << 0 << endl;
	// 	print_skinny_matrix(I_skinny_reordered, local_size);
	// 	cout << "I_skinny" << endl;
	// 	print_skinny_matrix(I_skinny, local_size);
	// }
	// MPI_Barrier(MPI_COMM_WORLD);
	// if (rank == 1) {
	// 	cout << "I_skinny_reordered, rank: " << 1 << endl;
	// 	print_skinny_matrix(I_skinny_reordered, local_size);
	// 	cout << "I_skinny" << endl;
	// 	print_skinny_matrix(I_skinny, local_size);
	// }
	// MPI_Barrier(MPI_COMM_WORLD);
	// for (int i = 0; i < rank_send.size(); i++) {
	// 	cout << "rank: " << rank << " to: " << i << "; ";
	// 	print_vector_i(rank_send[i], rank_send[i].size());
	// 	cout << endl;
	// }
	// for (int i = 0; i < rank_recv.size(); i++) {
	// 	cout << "rank: " << rank << " from: " << i << "; ";
	// 	print_vector_i(rank_recv[i], rank_recv[i].size());
	// 	cout << endl;
	// }

	if (rank == 0){
		v_old[0] = 1;
	}
	double start_time, end_time;
	MPI_Barrier(MPI_COMM_WORLD);
	start_time = MPI_Wtime();

	std::vector<std::vector<double> > send_buffer(size);
	std::vector<std::vector<double> > recv_buffer(size);
	for (int dest = 0; dest < size; dest++) {
		if (dest == rank) { continue; }
		if (rank_send[dest].size() == 0) { continue; }
		send_buffer[dest].resize(rank_send[dest].size());
		recv_buffer[dest].resize(rank_recv[dest].size());
	}

	for (int k = 0; k < reps; k++) {
		for (int i = 0; i < local_size; i++) {
			v_new[i] = 0.0;
			for (int j = 0; j < 4; j++) {
				v_new[i] += A_skinny[i][j] * v_old[I_skinny[i][j]];
			}
		}


		for (int dest = 0; dest < size; dest++) {
			if (dest == rank) { continue; }
			if (rank_send[dest].size() == 0) { continue; }
			for (int j = 0; j < rank_send[dest].size(); j++) {
				send_buffer[dest][j] = v_new[rank_send[dest][j]];
			}
		}

		int send_amount = 0;
		for (int dest = 0; dest < size; dest++) {
			if (dest == rank) { continue; }
			if (rank_send[dest].size() == 0) { continue; }
			send_amount++;
		}

		MPI_Request send_requests[send_amount];
		MPI_Request recv_requests[send_amount];

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
		
		for (int dest = 0; dest < size; dest++) {
			if (dest == rank) { continue; }
			if (rank_send[dest].size() == 0) { continue; }
			for (int j = 0; j < send_res_mat[rank][dest].size(); j++) {
				v_new[rank_recv[dest][j]] = recv_buffer[dest][j];
			}
		}

		std::swap(v_new, v_old);
	}

	
	// make the final vector v_fin
	// rank 0 receives from all other ranks
	std::vector<double> v_fin(n);
	if (rank == 0) {
		for (int i = 0; i < local_size; i++) {
			v_fin[i] = v_old[i];
		}
		for (int i = 1; i < size; i++) {
			MPI_Recv(v_fin.data() + i * local_size, local_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}
	// all other ranks send to rank 0
	else {
		MPI_Send(v_old.data(), local_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	end_time = MPI_Wtime();
	if (rank == 0) {
		cout << end_time - start_time << endl;
	}
	// print the final vector
	// if (rank == 0) {
	// 	cout << 1 << endl;
	// 	print_matrix(v_fin, n, a);
	// }
	MPI_Finalize();
	return 0;
}