#include <iostream>
#include <vector>
#include <mpi.h>
#include <numeric>
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

    const int shift = 2;
    const int a = 1 << shift;
    const int n = 2 * a * a;
    int reps = atoi(argv[1]);
    int local_size = n / size;
    int start_idx = rank * local_size;
    int end_idx = start_idx + local_size;
    std::vector<std::vector<double> > A_skinny(local_size, std::vector<double>(4));
    std::vector<std::vector<int> > I_skinny(local_size, std::vector<int>(4));
	std::vector<std::vector<std::vector<int> > > send_res_mat(size, std::vector<std::vector<int> >(size, std::vector<int>()));
    std::vector<double> v_old(local_size);
    std::vector<double> v_new(local_size);

	// Populates I_skinny and send_res_mat
    for (int i = start_idx; i < end_idx; i++) {
		int idx0 = 0 <= i - 1 && i % (2 * a) != 0 ? (i - 1) : i;
		int idx1 = i;
		int idx2 = i + 1 <= n && (i + 1) % (2 * a) != 0 ? (i + 1) : i;
		int idx3;
		if (i % 2 == 0){ idx3 = 2 * a + i + 1 <= n ? (2 * a + i + 1) : i; }
		else { idx3 = 0 <= i - 2 * a - 1 ? (i - 2 * a - 1) : i; }
        I_skinny[i - start_idx][0] = idx0;
		I_skinny[i - start_idx][1] = idx1;
		I_skinny[i - start_idx][2] = idx2;
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
	std::vector<std::vector<int> > I_skinny_reordered(local_size, std::vector<int>(4));
	std::map<int, int> rank_map;
	int next_idx = 0;
	int prev_idx = local_size;
	for (int i = 0; i < send_res_mat[rank].size(); i++) {
		if (i == rank) { continue; }
		for (int j = 0; j < send_res_mat[i][rank].size(); j++) {
			v_new.push_back(0.0);
			rank_send[i].push_back(send_res_mat[rank][i][j] % local_size);
			rank_recv[i].push_back(prev_idx);
			rank_map[send_res_mat[i][rank][j]] = prev_idx;
			prev_idx++;
		}
		for (int j = 0; j < send_res_mat[rank][i].size(); j++) {
		}
	}
	for (int i = 0; i < I_skinny_reordered.size(); i++) {
		I_skinny_reordered[i][0] = I_skinny[i][0] % local_size;
		I_skinny_reordered[i][1] = I_skinny[i][1] % local_size;
		I_skinny_reordered[i][2] = I_skinny[i][2] % local_size;
		if (I_skinny[i][3] >= start_idx && I_skinny[i][3] < end_idx){
			I_skinny_reordered[i][3] = I_skinny[i][3] % local_size;
		} 
		else {
			I_skinny_reordered[i][3] = rank_map[I_skinny[i][3]];

		}
	}
	
	// if (rank == 0) {
	// 	cout << "I_skinny_reordered" << endl;
	// 	print_skinny_matrix(I_skinny_reordered, local_size);
	// 	cout << "I_skinny" << endl;
	// 	print_skinny_matrix(I_skinny, local_size);
	// }
	// MPI_Barrier(MPI_COMM_WORLD);
	// if (rank == 1) {
	// 	cout << "I_skinny_reordered" << endl;
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

	
	MPI_Barrier(MPI_COMM_WORLD);
	v_old.resize(v_new.size());
	if (rank == 0) {
		v_old[0] = 1;
	}

	for (int k = 0; k < reps; k++) {
        for (int i = 0; i < local_size; i++) {
            v_new[i] = 0.0;
            for (int j = 0; j < 4; j++) {
                v_new[i] += A_skinny[i][j] * v_old[I_skinny[i][j]];
            }
        }

		MPI_Barrier(MPI_COMM_WORLD);
		
		for (int dest = 0; dest < size; dest++) {

			MPI_Request send_request;
			MPI_Request recv_request;

			for (int i = 0; i < rank_send[dest].size(); i++) {
				cout << "rank: " << rank << " sending to: " << dest << " value: " << v_new[rank_send[dest][i]] << " at index: " << rank_send[dest][i] << endl;
				MPI_Isend(&v_new[rank_send[dest][i]], 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &send_request);
			}
			for (int i = 0; i < rank_recv[dest].size(); i++) {
				MPI_Irecv(&v_new[rank_recv[dest][i]], 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &recv_request);
				cout << "rank: " << rank << " receiving from: " << dest << " value: " << v_new[rank_recv[dest][i]] << " at index: " << rank_recv[dest][i] << endl;
			}

		}
		std::swap(v_new, v_old);
	}

	MPI_Allgather(v_old.data() + start_idx, local_size, MPI_DOUBLE, v_old.data(), local_size, MPI_DOUBLE, MPI_COMM_WORLD);
	
	if (rank == 0) {
		print_matrix(v_old, n, 2 * a);
	}
	
	MPI_Finalize();
	return 0;
}