#include <iostream>
#include <vector>
#include <mpi.h>
#include <numeric>

using namespace std;

void print_skinny_matrix(const std::vector<std::vector<int> >& skinny,const int& n) {
	for (int i = 0; i < n; i++) {
		cout << skinny[i][0] << " " << skinny[i][1] << " " << skinny[i][2] << " " << skinny[i][3] << endl;
	}
}

void print_matrix(const std::vector<double>& v,const int& n,const int& a) {
    for (int i = 0; i < n; i++) {
        cout << v[i] << " ";
        if (i % a == a - 1) {
            cout << endl;
        }
    }
}

void print_vector(const std::vector<double>& v,const int& n) {
	for (int i = 0; i < n; i++) {
		cout << v[i] << " ";
	}
}


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
    std::vector<std::vector<double> > A_skinny(local_size, std::vector<double>(4));
    std::vector<std::vector<int> > I_skinny(local_size, std::vector<int>(4));
    std::vector<double> v_old(n);
    std::vector<double> v_new(n);
	std::vector<double> local_separator;
	std::vector<double> non_separator;
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
		if ((start_idx > idx0 || idx0 > end_idx) ||
			(start_idx > idx1 || idx1 > end_idx) ||
			(start_idx > idx2 || idx2 > end_idx) ||
			(start_idx > idx3 || idx3 > end_idx)) {
			local_separator.push_back(i);
		}
		else {
			non_separator.push_back(i);
		}
    }
	cout << "Rank " << rank << " separator: ";
	print_vector(local_separator, local_separator.size());
	cout << endl;
	// Initialize a new vector for the reordered I_skinny matrix
	std::vector<std::vector<int> > I_skinny_reordered(local_size, std::vector<int>(4));

	// Copy the separator values to the reordered matrix
	for (int i = 0; i < local_separator.size(); i++) {
		I_skinny_reordered[i] = I_skinny[local_separator[i] - start_idx];
		
	}

	// Copy the non-separator values to the reordered matrix
	for (int i = 0; i < non_separator.size(); i++) {
		I_skinny_reordered[i+local_separator.size()] = I_skinny[non_separator[i - start_idx]];
		
	}

	// Replace the original I_skinny matrix with the reordered matrix
	I_skinny = I_skinny_reordered;


	std::vector<int> sep_sizes(size);
	int sep_size = local_separator.size();
	MPI_Allgather(&sep_size, 1, MPI_INT, sep_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
	std::vector<std::vector<double> > separator(size);
	for (int i = 0; i < size; i++) {
		separator[i].resize(sep_sizes[i]);
	}
    // Fill each rank's portion of the matrix

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
	for (int k = 0; k < reps; k++) {
		for (int i = 0; i < local_size; i++) {
			v_new[i + start_idx] = 0;
			for (int j = 0; j < 4; j++) {
				v_new[i + start_idx] += A_skinny[i][j] * v_old[I_skinny[i][j]];;
			}
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		for (int i = 0; i < size; i++){
			MPI_Bcast(v_new.data() + start_idx, sep_sizes[i], MPI_DOUBLE, i, MPI_COMM_WORLD);
		}
		


		MPI_Barrier(MPI_COMM_WORLD);
		std::swap(v_new, v_old);
    }
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allgather(v_old.data() + start_idx, local_size, MPI_DOUBLE, v_old.data(), local_size, MPI_DOUBLE, MPI_COMM_WORLD);
	std::vector<double> v_old_reordered(n);
	// for (int i = 0; i < local_separator.size(); i++) {
	// 	v_old_reordered[i] = v_old[local_separator[i]];
	// }
	// for (int i = 0; i < non_separator.size(); i++) {
	// 	v_old_reordered[i + local_separator.size()] = v_old[non_separator[i]];
	// }
	if (rank == 0) {
		print_matrix(v_old, n, a); // Update the print_vector call
		cout << endl;
	}

    MPI_Finalize();

    return 0;
}
