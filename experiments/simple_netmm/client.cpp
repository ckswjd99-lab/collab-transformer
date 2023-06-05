#include "config.hpp"
#include "worker.hpp"

void error_handling(char *message)
{
        fputs(message, stderr);
        fputc('\n', stderr);
        exit(1);
}

int main(int argc, char **argv) {

    // init semaphores
    init_sem();

    // create workers & sender
    pthread_t workers_thread[WORKER_NUM];
    pthread_t sender_thread;
    pthread_t receiver_thread;
    int thread_id;
    void *thread_status;

    for (int i=0; i<WORKER_NUM; i++) {
        int *worker_id = (int *)malloc(sizeof(int));
        *worker_id = i;
        if ((thread_id = pthread_create(&workers_thread[i], NULL, worker, worker_id)) < 0) {
            perror("pthread_create fail!\n");
            exit(EXIT_FAILURE);
        }
    }


    // connect to server
    int sock;
    struct sockaddr_in serv_addr;
    int str_len;

    if(argc != 3) {
        printf("Usage : %s <IP> <port> \n", argv[0]);
        exit(1);
    }

    sock = socket(PF_INET, SOCK_STREAM, 0);
    if(sock == -1)
        error_handling("socket() error");

    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(argv[1]);
    serv_addr.sin_port = htons(atoi(argv[2]));

    if( connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) == -1)
        error_handling("connect() error");

    // start work!
    int seed = 0;

    if ((thread_id = pthread_create(&sender_thread, NULL, sender, (void *)&sock)) < 0) {
        perror("pthread_create fail!\n");
        exit(EXIT_FAILURE);
    }
    if ((thread_id = pthread_create(&receiver_thread, NULL, receiver, (void *)&sock)) < 0) {
        perror("pthread_create fail!\n");
        exit(EXIT_FAILURE);
    }

    for (int i=0; i<HEAD_NUM/DEVICE_NUM; i++) {
        push_workqueue(new Matrix(ENCO_QKV, i));
    }
    for (int i=0; i<SPLIT_NUM/DEVICE_NUM; i++) {
        push_workqueue(new Matrix(ENCO_DUMMY, i));
    }

    for (int i=0; i<WORKER_NUM; i++) {
        push_workqueue(NULL);
    }

    for (int i=0; i<WORKER_NUM; i++) {
        pthread_join(workers_thread[i], (void**)&thread_status);
    }

    push_sendqueue(NULL);

    pthread_join(sender_thread, (void **)&thread_status);
    pthread_join(receiver_thread, (void **)&thread_status);

    close(sock);


    return 0;
}