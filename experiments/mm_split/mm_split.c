#include <cblas.h>

#define M   64
#define N   768
#define K   768

#define SPLIT_N 12
#define REPEAT_N 128

float embedded  [K * M];
float W_QKV     [3*N * K];
float QKV       [3*N * M];

int main(int argc, char** argv) {

    for (int r=0; r<REPEAT_N; r++) {
        for (int i=0; i<SPLIT_N; i++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, 3*N/SPLIT_N, K, 1.0, embedded, K, W_QKV, 3*N/SPLIT_N, 1.0, QKV, 3*N/SPLIT_N);
        }
    }

    return 0;
}