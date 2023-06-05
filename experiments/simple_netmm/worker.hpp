#ifndef __WORKER_HPP
#define __WORKER_HPP

#include "config.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <pthread.h>

void push_workqueue(Matrix *mat);

Matrix *pop_workqueue();

Matrix *find_workqueue(EncoderLayer layer, int idx);

void remove_workqueue(EncoderLayer layer, int idx);

void push_sendqueue(Matrix *mat);

Matrix *pop_sendqueue();

void *worker(void *args);
void *worker_solo(void *args);

void *sender(void *args);

void *receiver(void *args);

void init_sem();


#endif