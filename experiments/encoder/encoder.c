#include <cblas.h>

#define TOKEN_N     128
#define EMBED_N     768
#define HEAD_N      12
#define SHEAD_N     (EMBED_N/HEAD_N)

/* Intermediates */
float input_token[TOKEN_N * EMBED_N];
float Q[TOKEN_N * SHEAD_N * HEAD_N];
float K[TOKEN_N * SHEAD_N * HEAD_N];
float V[TOKEN_N * SHEAD_N * HEAD_N];
float QK_T[TOKEN_N * TOKEN_N * HEAD_N];
float SHA[TOKEN_N * SHEAD_N * HEAD_N];
float MHA[TOKEN_N * EMBED_N];
float FFN1[TOKEN_N * 4*EMBED_N];
float FFN2[4*TOKEN_N * EMBED_N];

/* Weights */
float W_Q[EMBED_N * EMBED_N];
float W_K[EMBED_N * EMBED_N];
float W_V[EMBED_N * EMBED_N];
float W_O[EMBED_N * EMBED_N];
float W_FFN1[EMBED_N * (4*EMBED_N)];
float W_FFN2[(4*EMBED_N) * EMBED_N];

int main(int argc, char** argv) {

    /* Calculate QKV */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, TOKEN_N, EMBED_N, EMBED_N, 1.0, input_token, EMBED_N, W_Q, EMBED_N, 1.0, Q, EMBED_N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, TOKEN_N, EMBED_N, EMBED_N, 1.0, input_token, EMBED_N, W_K, EMBED_N, 1.0, K, EMBED_N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, TOKEN_N, EMBED_N, EMBED_N, 1.0, input_token, EMBED_N, W_V, EMBED_N, 1.0, V, EMBED_N);

    for (int i=0; i<HEAD_N; i++) {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans, 
            TOKEN_N, TOKEN_N, SHEAD_N, 
            1.0, 
            Q + i*(TOKEN_N * SHEAD_N), SHEAD_N, 
            K + i*(TOKEN_N * SHEAD_N), SHEAD_N, 
            1.0, 
            QK_T + i*(TOKEN_N * TOKEN_N), TOKEN_N
        );
    }

    // softmax (omitted)

    for (int i=0; i<HEAD_N; i++) {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            TOKEN_N, SHEAD_N, TOKEN_N,
            1.0,
            QK_T + i*(TOKEN_N * TOKEN_N), TOKEN_N,
            V + i*(TOKEN_N * SHEAD_N), SHEAD_N,
            1.0,
            SHA + i*(TOKEN_N * SHEAD_N), SHEAD_N
        );
    }

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        TOKEN_N, EMBED_N, EMBED_N,
        1.0,
        SHA, EMBED_N,
        W_O, EMBED_N,
        1.0,
        MHA, EMBED_N
    );

    // add & normalize

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        TOKEN_N, 4*EMBED_N, EMBED_N,
        1.0,
        MHA, EMBED_N,
        W_FFN1, 4*EMBED_N,
        1.0,
        FFN1, 4*EMBED_N
    );

    // GeLU

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        TOKEN_N, EMBED_N, 4*EMBED_N,
        1.0,
        FFN1, 4*EMBED_N,
        W_FFN2, EMBED_N,
        1.0,
        FFN2, EMBED_N
    );

    


    return 0;
}