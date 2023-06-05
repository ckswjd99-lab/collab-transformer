#include "config.hpp"
#include "worker.hpp"


void error_handling(char *message)
{
        fputs(message, stderr);
        fputc('\n', stderr);
        exit(1);
}

int main(int argc, char **argv) {

    // create server
    int serv_sock;
    int clnt_sock;
    int str_len;

    struct sockaddr_in serv_addr;
    struct sockaddr_in clnt_addr;
    unsigned int clnt_addr_size;

    pthread_t send_thread, recv_thread;
    int status;

    if(argc != 2) {
        printf("Usage : &s <port>\n", argv[0]);
        exit(1);
    }

    serv_sock = socket(PF_INET, SOCK_STREAM, 0);
    if(serv_sock == -1)
        error_handling("socket() error");

    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(atoi(argv[1]));

    
    if( bind(serv_sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr) )==-1)
        error_handling("bind() error");

    if(listen(serv_sock, 5) == -1)
        error_handling("listen() error");

    clnt_addr_size = sizeof(clnt_addr);


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

    // listen to client
    clnt_sock = accept(serv_sock, (struct sockaddr*)&clnt_addr, &clnt_addr_size);
    if(clnt_sock == -1)
        error_handling("accept() error");

    if ((thread_id = pthread_create(&sender_thread, NULL, sender, (void *)&clnt_sock)) < 0) {
        perror("pthread_create fail!\n");
        exit(EXIT_FAILURE);
    }
    if ((thread_id = pthread_create(&receiver_thread, NULL, receiver, (void *)&clnt_sock)) < 0) {
        perror("pthread_create fail!\n");
        exit(EXIT_FAILURE);
    }

    // start work!
    int seed = 1;

    for (int i=HEAD_NUM/DEVICE_NUM; i<HEAD_NUM; i++) {
        push_workqueue(new Matrix(ENCO_QKV, i));
    }
    for (int i=SPLIT_NUM/DEVICE_NUM; i<SPLIT_NUM; i++) {
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

    close(clnt_sock);
    close(serv_sock);

    return 0;
}