#include "config.hpp"
#include "worker.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <pthread.h>
#include <time.h>

#define DLOG(worker_idx, event, task, task_idx, timestamp) (\
    dprintf(1, "{\"worker\": %d, \"event\": \"%s\", \"task\": \"%s\", \"index\": %d, \"timestamp\": %ld},\n", worker_idx, event, task, task_idx, timestamp)\
)

sem_t access_workqueue_sem;
sem_t pop_workqueue_sem;
sem_t push_workqueue_sem;

std::vector<Matrix *> workqueue;

sem_t access_sendqueue_sem;
sem_t pop_sendqueue_sem;
sem_t push_sendqueue_sem;

std::vector<Matrix *> sendqueue;

sem_t access_socket_sem;


float embedded  [TOKEN_NUM * EMBED_CHAN];
float W_Q       [EMBED_CHAN * EMBED_CHAN];
float W_K       [EMBED_CHAN * EMBED_CHAN];
float W_V       [EMBED_CHAN * EMBED_CHAN];
float Q         [TOKEN_NUM * EMBED_CHAN];
float K         [TOKEN_NUM * EMBED_CHAN];
float V         [TOKEN_NUM * EMBED_CHAN];
float QK_T      [HEAD_NUM * TOKEN_NUM * TOKEN_NUM];
float SHA       [TOKEN_NUM * EMBED_CHAN];

float DUM_A     [DUMMY_M * DUMMY_K];
float DUM_B     [DUMMY_K * DUMMY_N];
float DUM_C     [DUMMY_M * DUMMY_N];

unsigned int write_n(int __fd, const void *__buf, size_t __n) {
    int total_write = 0;
    int now_write;
    while(total_write < __n) {
        now_write = write(__fd, __buf, __n - total_write);
        if(now_write < 0) continue;
        total_write += now_write;
        __buf = __buf + now_write;
        
    }

    return total_write;
}

unsigned int read_n(int __fd, void *__buf, size_t __nbytes) {
    int total_read = 0;
    int now_read;
    while(total_read < __nbytes) {
        now_read = read(__fd, __buf, __nbytes - total_read);
        if(now_read < 0) continue;
        total_read += now_read;
        __buf = __buf + now_read;
    }

    return total_read;
}


void push_workqueue(Matrix *mat) {
    sem_wait(&push_workqueue_sem);
    sem_wait(&access_workqueue_sem);

    workqueue.push_back(mat);

    sem_post(&access_workqueue_sem);
    sem_post(&pop_workqueue_sem);
}

Matrix *pop_workqueue() {
    sem_wait(&pop_workqueue_sem);
    sem_wait(&access_workqueue_sem);

    Matrix *result;

    result = workqueue.at(0);
    workqueue.erase(workqueue.begin());

    sem_post(&access_workqueue_sem);
    sem_post(&push_workqueue_sem);

    return result;
}

Matrix *find_workqueue(EncoderLayer layer, int idx) {
    Matrix *result = NULL;

    sem_wait(&access_workqueue_sem);

    std::vector<Matrix *>::iterator iter;
    for(iter = workqueue.begin(); iter != workqueue.end(); iter++) {
        if((*iter)->layer == layer && (*iter)->idx == idx) {
            result = *iter;
            break;
        }
    }

    sem_post(&access_workqueue_sem);

    return result;
}

void remove_workqueue(EncoderLayer layer, int idx) {
    sem_wait(&access_workqueue_sem);

    std::vector<Matrix *>::iterator iter;
    for(iter = workqueue.begin(); iter != workqueue.end(); iter++) {
        if((*iter)->layer == layer && (*iter)->idx == idx) {
            workqueue.erase(iter);
            break;
        }
    }

    sem_post(&access_workqueue_sem);
}

void push_sendqueue(Matrix *mat) {
    sem_wait(&push_sendqueue_sem);
    sem_wait(&access_sendqueue_sem);

    sendqueue.push_back(mat);
    
    sem_post(&access_sendqueue_sem);
    sem_post(&pop_sendqueue_sem);
}

Matrix *pop_sendqueue() {
    sem_wait(&pop_sendqueue_sem);
    sem_wait(&access_sendqueue_sem);

    Matrix *result;

    if (sendqueue.size() == 0) result = NULL;
    else {
        result = sendqueue.at(0);
        sendqueue.erase(sendqueue.begin());
    }
    
    sem_post(&access_sendqueue_sem);
    sem_post(&push_sendqueue_sem);

    return result;
}

void *worker(void *args) {
    int worker_id = *(int *)args;
    // dprintf(1, "worker %d start working!\n", worker_id);

    Matrix *task;
    while (task = pop_workqueue()) {
        // if (task->origin == Q) dprintf(1, "worker %d fetched Q[%d] (%d) left\n", worker_id, task->idx, workqueue.size());
        // if (task->origin == K) dprintf(1, "worker %d fetched K[%d] (%d) left\n", worker_id, task->idx, workqueue.size());
        // if (task->origin == V) dprintf(1, "worker %d fetched V[%d] (%d) left\n", worker_id, task->idx, workqueue.size());

        EncoderLayer layer = task->layer;

        if (layer == ENCO_QKV) {
            #if __DEBUG
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            DLOG(worker_id, "start", "QKV", task->idx, ts.tv_sec * 1000000000 + ts.tv_nsec);
            #endif

            float *W_Q_idx = W_Q + (EMBED_CHAN / HEAD_NUM * task->idx);
            float *W_K_idx = W_K + (EMBED_CHAN / HEAD_NUM * task->idx);
            float *W_V_idx = W_V + (EMBED_CHAN / HEAD_NUM * task->idx);
            float *Q_idx = Q + (EMBED_CHAN / HEAD_NUM * task->idx);
            float *K_idx = K + (EMBED_CHAN / HEAD_NUM * task->idx);
            float *V_idx = V + (EMBED_CHAN / HEAD_NUM * task->idx);
            float *QKT_idx = QK_T + (TOKEN_NUM * TOKEN_NUM * task->idx);
            float *SHA_idx = QK_T + (EMBED_CHAN / HEAD_NUM * task->idx);
            
            // gen Q
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                TOKEN_NUM, HEAD_NUM, EMBED_CHAN,
                1.0,
                embedded, EMBED_CHAN,
                W_Q_idx, EMBED_CHAN,
                0.0,
                Q_idx, EMBED_CHAN
            );
            
            // gen K
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                TOKEN_NUM, HEAD_NUM, EMBED_CHAN,
                1.0,
                embedded, EMBED_CHAN,
                W_K_idx, EMBED_CHAN,
                0.0,
                K_idx, EMBED_CHAN
            );
            
            // gen V
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                TOKEN_NUM, HEAD_NUM, EMBED_CHAN,
                1.0,
                embedded, EMBED_CHAN,
                W_V_idx, EMBED_CHAN,
                0.0,
                V_idx, EMBED_CHAN
            );

            // #if __DEBUG
            // clock_gettime(CLOCK_REALTIME, &ts);
            // DLOG(worker_id, "qkvgened", "QKV", task->idx, ts.tv_sec * 1000000000 + ts.tv_nsec);
            // #endif

            // gen Attention
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,
                TOKEN_NUM, TOKEN_NUM, EMBED_CHAN / HEAD_NUM,
                1.0,
                Q_idx, EMBED_CHAN,
                K_idx, EMBED_CHAN,
                0.0,
                QKT_idx, TOKEN_NUM
            );

            // #if __DEBUG
            // clock_gettime(CLOCK_REALTIME, &ts);
            // DLOG(worker_id, "attgened", "QKV", task->idx, ts.tv_sec * 1000000000 + ts.tv_nsec);
            // #endif

            // do SOFTMAX

            // do sm(QK^T)V
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                TOKEN_NUM, EMBED_CHAN / HEAD_NUM, TOKEN_NUM,
                1.0,
                QKT_idx, TOKEN_NUM,
                V_idx, EMBED_CHAN,
                0.0,
                SHA_idx, EMBED_CHAN
            );

            #if __DEBUG
            clock_gettime(CLOCK_REALTIME, &ts);
            DLOG(worker_id, "end", "QKV", task->idx, ts.tv_sec * 1000000000 + ts.tv_nsec);
            #endif


            push_sendqueue(task);
        }
        else if (layer == ENCO_DUMMY) {
            #if __DEBUG
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            DLOG(worker_id, "start", "DUMMY", task->idx, ts.tv_sec * 1000000000 + ts.tv_nsec);
            #endif

            float *dA_idx = DUM_A + task->idx * (DUMMY_M / SPLIT_NUM) * DUMMY_K;
            float *dB_idx = DUM_B;
            float *dC_idx = DUM_C + task->idx * (DUMMY_M / SPLIT_NUM) * DUMMY_N;

            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (DUMMY_M / SPLIT_NUM), DUMMY_N, DUMMY_K,
                1.0,
                dA_idx, DUMMY_K,
                dB_idx, DUMMY_N,
                0.0,
                dC_idx, DUMMY_N
            );

            #if __DEBUG
            clock_gettime(CLOCK_REALTIME, &ts);
            DLOG(worker_id, "end", "DUMMY", task->idx, ts.tv_sec * 1000000000 + ts.tv_nsec);
            #endif
        }
    }

    return NULL;
}

void *worker_solo(void *args) {
    int worker_id = *(int *)args;
    // dprintf(1, "worker %d start working!\n", worker_id);

    Matrix *task;
    while (task = pop_workqueue()) {
        // if (task->origin == Q) dprintf(1, "worker %d fetched Q[%d] (%d) left\n", worker_id, task->idx, workqueue.size());
        // if (task->origin == K) dprintf(1, "worker %d fetched K[%d] (%d) left\n", worker_id, task->idx, workqueue.size());
        // if (task->origin == V) dprintf(1, "worker %d fetched V[%d] (%d) left\n", worker_id, task->idx, workqueue.size());

        EncoderLayer layer = task->layer;

        if (layer == ENCO_QKV) {
            #if __DEBUG
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            DLOG(worker_id, "start", "QKV", task->idx, ts.tv_sec * 1000000000 + ts.tv_nsec);
            #endif

            float *W_Q_idx = W_Q + (EMBED_CHAN / HEAD_NUM * task->idx);
            float *W_K_idx = W_K + (EMBED_CHAN / HEAD_NUM * task->idx);
            float *W_V_idx = W_V + (EMBED_CHAN / HEAD_NUM * task->idx);
            float *Q_idx = Q + (EMBED_CHAN / HEAD_NUM * task->idx);
            float *K_idx = K + (EMBED_CHAN / HEAD_NUM * task->idx);
            float *V_idx = V + (EMBED_CHAN / HEAD_NUM * task->idx);
            float *QKT_idx = QK_T + (TOKEN_NUM * TOKEN_NUM * task->idx);
            float *SHA_idx = QK_T + (EMBED_CHAN / HEAD_NUM * task->idx);
            
            // gen Q
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                TOKEN_NUM, HEAD_NUM, EMBED_CHAN,
                1.0,
                embedded, EMBED_CHAN,
                W_Q_idx, EMBED_CHAN,
                0.0,
                Q_idx, EMBED_CHAN
            );
            
            // gen K
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                TOKEN_NUM, HEAD_NUM, EMBED_CHAN,
                1.0,
                embedded, EMBED_CHAN,
                W_K_idx, EMBED_CHAN,
                0.0,
                K_idx, EMBED_CHAN
            );
            
            // gen V
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                TOKEN_NUM, HEAD_NUM, EMBED_CHAN,
                1.0,
                embedded, EMBED_CHAN,
                W_V_idx, EMBED_CHAN,
                0.0,
                V_idx, EMBED_CHAN
            );

            // gen Attention
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,
                TOKEN_NUM, TOKEN_NUM, EMBED_CHAN / HEAD_NUM,
                1.0,
                Q_idx, EMBED_CHAN,
                K_idx, EMBED_CHAN,
                0.0,
                QKT_idx, TOKEN_NUM
            );

            // do SOFTMAX

            // do sm(QK^T)V
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                TOKEN_NUM, EMBED_CHAN / HEAD_NUM, TOKEN_NUM,
                1.0,
                QKT_idx, TOKEN_NUM,
                V_idx, EMBED_CHAN,
                0.0,
                SHA_idx, EMBED_CHAN
            );

            #if __DEBUG
            clock_gettime(CLOCK_REALTIME, &ts);
            DLOG(worker_id, "end", "QKV", task->idx, ts.tv_sec * 1000000000 + ts.tv_nsec);
            #endif

        }
        else if (layer == ENCO_DUMMY) {
            #if __DEBUG
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            DLOG(worker_id, "start", "DUMMY", task->idx, ts.tv_sec * 1000000000 + ts.tv_nsec);
            #endif

            float *dA_idx = DUM_A + task->idx * (DUMMY_M / SPLIT_NUM) * DUMMY_K;
            float *dB_idx = DUM_B;
            float *dC_idx = DUM_C + task->idx * (DUMMY_M / SPLIT_NUM) * DUMMY_N;

            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (DUMMY_M / SPLIT_NUM), DUMMY_N, DUMMY_K,
                1.0,
                dA_idx, DUMMY_K,
                dB_idx, DUMMY_N,
                0.0,
                dC_idx, DUMMY_N
            );

            #if __DEBUG
            clock_gettime(CLOCK_REALTIME, &ts);
            DLOG(worker_id, "end", "DUMMY", task->idx, ts.tv_sec * 1000000000 + ts.tv_nsec);
            #endif
        }


    }

    return NULL;
}

void *sender(void *args) {
    int client_sock = *(int *)args;
    Matrix *task;
    while (task = pop_sendqueue()) {
        #if __DEBUG
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        DLOG(WORKER_NUM, "start", "SEND", task->idx, ts.tv_sec * 1000000000 + ts.tv_nsec);
        #endif

        write_n(client_sock, &(task->idx), sizeof(int));

        float *data = SHA + task->idx * EMBED_CHAN / HEAD_NUM;

        // for (int i=0; i<TOKEN_NUM; i++) {
            write_n(client_sock, SHA, sizeof(float) * EMBED_CHAN / HEAD_NUM * TOKEN_NUM / DEVICE_NUM);
            data += EMBED_CHAN;
        // }

        #if __DEBUG
        clock_gettime(CLOCK_REALTIME, &ts);
        DLOG(WORKER_NUM, "end", "SEND", task->idx, ts.tv_sec * 1000000000 + ts.tv_nsec);
        #endif

    }
    #if __DEBUG
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    DLOG(WORKER_NUM, "terminate", "", -1, ts.tv_sec * 1000000000 + ts.tv_nsec);
    #endif
    int terminate = -1;
    write_n(client_sock, &terminate, sizeof(int));
    return NULL;
}

void *receiver(void *args) {
    int client_sock = *(int *)args;

    int idx;
    float *data;
    while(1) {
        read_n(client_sock, &idx, sizeof(int));
        if (idx < 0) {
            #if __DEBUG
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            DLOG(WORKER_NUM+1, "terminate", "", -1, ts.tv_sec * 1000000000 + ts.tv_nsec);
            #endif
            break;
        }

        #if __DEBUG
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        DLOG(WORKER_NUM+1, "start", "READ", idx, ts.tv_sec * 1000000000 + ts.tv_nsec);
        #endif

        data = SHA + idx * EMBED_CHAN / HEAD_NUM;

        // for (int i=0; i<TOKEN_NUM; i++) {
            read_n(client_sock, SHA, sizeof(float) * EMBED_CHAN / HEAD_NUM * TOKEN_NUM / DEVICE_NUM);
            data += EMBED_CHAN;
        // }

        #if __DEBUG
        clock_gettime(CLOCK_REALTIME, &ts);
        DLOG(WORKER_NUM+1, "end", "READ", idx, ts.tv_sec * 1000000000 + ts.tv_nsec);
        #endif

    }

    return NULL;
}

void init_sem() {
    sem_init(&access_workqueue_sem, 0, 1);
    sem_init(&push_workqueue_sem, 0, WORKER_NUM);
    sem_init(&pop_workqueue_sem, 0, 0);

    sem_init(&access_sendqueue_sem, 0, 1);
    sem_init(&push_sendqueue_sem, 0, WORKER_NUM);
    sem_init(&pop_sendqueue_sem, 0, 0);

    sem_init(&access_socket_sem, 0, 1);
}