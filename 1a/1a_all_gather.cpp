// void find_neigbors(
// 	std::vector<std::vector<int> >& I_skinny,
// 	const int& n,
// 	const int& a) {
// 	for (int i = 0; i < n; i++) {
// 		I_skinny[i][0] = 0 <= i-1 && i%a != 0 ? i-1 : i;
// 		I_skinny[i][1] = i;
// 		I_skinny[i][2] = i+1 <= n && (i+1)%a != 0 ? i+1 : i;
// 		if (i % 2 == 0){
// 			I_skinny[i][3] = a+i+1 <= n ? a+i+1 : i;
// 		}
// 		else {
// 			I_skinny[i][3] = 0 <= i-a-1 ? i-a-1 : i;
// 		}
// 	}

// }


#include <iostream>
#include <vector>
#include <mpi.h>
#include <numeric>

using namespace std;

void matrix_multiply(
    const std::vector<std::vector<int> >& I_skinny,
    const std::vector<std::vector<double> >& A_skinny,
    std::vector<double>& v_old,
    std::vector<double>& v_new,
    const int& n,
    const int& steps,
    const int& start_idx,
    const int& end_idx) {
    for (int k = 0; k < steps; k++) {
        for (int i = start_idx; i < end_idx; i++) {
            v_new[i] = 0.0;
            for (int j = 0; j < 4; j++) {
                v_new[i] += A_skinny[i][j] * v_old[I_skinny[i][j]];
            }
        }
        std::swap(v_new, v_old);
    }
}

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

void print_vector(
	const std::vector<double>& v,
	const int& n) {
	for (int i = 0; i < n; i++) {
		cout << v[i] << " ";
	}
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int shift = 4;
    const int a = 1 << shift;
    const int n = 2 * a * a;
    int reps = atoi(argv[1]);
    int local_size = n / size;
    int start_idx = rank * local_size;
    int end_idx = start_idx + local_size;
	int sep_size = 0;
    std::vector<std::vector<double> > A_skinny(local_size, std::vector<double>(4));
    std::vector<std::vector<int> > I_skinny(local_size, std::vector<int>(4));
    std::vector<double> v_old(n);
    std::vector<double> v_new(n);
	std::vector<double> separator;
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
		if ((start_idx >= idx0 || idx0 >= end_idx) ||
			(start_idx >= idx1 || idx1 >= end_idx) ||
			(start_idx >= idx2 || idx2 >= end_idx) ||
			(start_idx >= idx3 || idx3 >= end_idx)) {
			separator.push_back(i);
		}

        // I_skinny[i - start_idx][0] = 0 <= i - 1 && i % (2 * a) != 0 ? (i - 1) : i;
        // I_skinny[i - start_idx][1] = i;
        // I_skinny[i - start_idx][2] = i + 1 <= n && (i + 1) % (2 * a) != 0 ? (i + 1) : i;
        // if (i % 2 == 0) {
        //     I_skinny[i - start_idx][3] = 2 * a + i + 1 <= n ? (2 * a + i + 1) : i;
        // } else {
        //     I_skinny[i - start_idx][3] = 0 <= i - 2 * a - 1 ? (i - 2 * a - 1) : i;
        // }
        	
		// if (start_idx > idx0 || idx0 > end_idx){ separator[i - start_idx] = i;}
		// if (start_idx > idx1 || idx1 > end_idx){ separator[i - start_idx] = i;}
		// if (start_idx > idx2 || idx2 > end_idx){ separator[i - start_idx] = i;}
		// if (start_idx > idx3 || idx3 > end_idx){ separator[i - start_idx] = i;}

    }
	// MPI_Gather(separator.data() + start_idx, local_size, MPI_DOUBLE, separator.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// print_vector(separator, separator.size());
	// cout << "end of separator on rank: " << rank << endl;

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
    
    // for (int i = 0; i < local_size; i++) {
    //     A_skinny[i][0] = 0.2;
    //     A_skinny[i][1] = 0.4;
    //     A_skinny[i][2] = 0.2;
    //     A_skinny[i][3] = 0.2;
    // }

    if (rank == 0) {
        v_old[0] = 1;
    }
    MPI_Bcast(v_old.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        cout << reps << endl;
    }
    for (int k = 0; k < reps; k++) {
        for (int i = 0; i < local_size; i++) {
            v_new[i + start_idx] = 0.0;
            for (int j = 0; j < 4; j++) {
                v_new[i + start_idx] += A_skinny[i][j] * v_old[I_skinny[i][j]];
            }
        }
        MPI_Allgather(v_new.data() + start_idx, local_size, MPI_DOUBLE, v_new.data(), local_size, MPI_DOUBLE, MPI_COMM_WORLD);
        std::swap(v_new, v_old);
        if (rank == 0) {
            print_matrix(v_old, n, 2 * a); // Update the print_vector call
        }
		//cout << std::accumulate(v_old.begin(), v_old.end(), 0.0) << endl;
	
    }

    MPI_Finalize();

    return 0;
}
