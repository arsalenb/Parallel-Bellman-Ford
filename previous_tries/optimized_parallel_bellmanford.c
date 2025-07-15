#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <limits.h>
#include <stdbool.h>

#define INFINITY INT_MAX

#ifdef _WIN32
#include <windows.h>
long long gettime(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (counter.QuadPart * 1000000000LL) / freq.QuadPart;
}
#else
#include <time.h>
long long gettime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000000000LL) + ts.tv_nsec;
}
#endif

typedef struct Graph {
    int V;
    int E;
    int** adjMatrix;
} Graph;

typedef struct {
    Graph* graph;
    int* d;
    int* d_local;
    int* mask;
    int* mask1;
    int* p;
    int* flag;
    int thread_id;
    int num_threads;
} ThreadData;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_barrier_t barrier;

Graph* createGraph(int V) {
    Graph* graph = malloc(sizeof(Graph));
    graph->V = V;
    graph->E = 0;
    graph->adjMatrix = malloc(V * sizeof(int*));
    for (int i = 0; i < V; i++) {
        graph->adjMatrix[i] = calloc(V, sizeof(int));
    }
    return graph;
}

void addEdge(Graph* graph, int u, int v, int w) {
    if (graph->adjMatrix[u][v] == 0) graph->E++;
    graph->adjMatrix[u][v] = w;
}

Graph* readGraph(char* filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Could not open file %s\n", filename);
        return NULL;
    }

    int V, E;
    fscanf(file, "%d\n%d\n", &E, &V);
    printf("Edges: %d, Vertices: %d\n", E, V);

    Graph* graph = createGraph(V);
    char line[2048];
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ":");
        int u = atoi(token);
        while ((token = strtok(NULL, ";"))) {
            int v, w;
            sscanf(token, "%d,%d", &v, &w);
            addEdge(graph, u, v, w);
        }
    }
    fclose(file);
    return graph;
}

void print_results(int* d, int* p, int V) {
    printf("Distance array: ");
    for (int i = 0; i < V; i++) {
        printf(d[i] == INFINITY ? "INF " : "%d ", d[i]);
    }
    printf("\nPredecessor array: ");
    for (int i = 0; i < V; i++) printf("%d ", p[i]);
    printf("\n");
}

void* parallel_init(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int first = (data->thread_id * data->graph->V) / data->num_threads;
    int last = ((data->thread_id + 1) * data->graph->V) / data->num_threads;

    for (int i = first; i < last; i++) {
        data->d[i] = INFINITY;
        data->d_local[i] = INFINITY;
        data->p[i] = -1;
        data->mask[i] = 0;
        data->mask1[i] = 0;
    }

    if (data->thread_id == 0) {
        data->d[0] = 0;
        data->d_local[0] = 0;
        data->mask[0] = 1;
    }

    pthread_exit(NULL);
}

void* bellmanford_parallel_helper(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    Graph* g = data->graph;
    int V = g->V;
    int first = (data->thread_id * V) / data->num_threads;
    int last = ((data->thread_id + 1) * V) / data->num_threads;
    
    for (int k = 0; k < V; k++) {
        for (int u = first; u < last; u++) {
            for (int v = 0; v < V; v++) {
                int w = g->adjMatrix[u][v];
                if (w && data->mask[u] && data->d[u] != INFINITY && 
                    data->d_local[v] > data->d[u] + w) {
                    data->d_local[v] = data->d[u] + w;
                    data->mask1[v] = 1;
                    data->p[v] = u;
                }
            }
        }
        
        pthread_barrier_wait(&barrier);
        
        for (int v = first; v < last; v++) {
            if (data->d_local[v] != data->d[v]) {
                data->d[v] = data->d_local[v];
                pthread_mutex_lock(&mutex);
                *(data->flag) = 1;
                pthread_mutex_unlock(&mutex);
            }
        }
        
        pthread_barrier_wait(&barrier);
        
        if (*(data->flag) == 0) {
            pthread_barrier_wait(&barrier);
            break;
        }
        
        pthread_barrier_wait(&barrier);
        
        if (data->thread_id == 0) *(data->flag) = 0;
        
        for (int v = first; v < last; v++) {
            data->mask[v] = data->mask1[v];
            data->mask1[v] = 0;
        }
        
        pthread_barrier_wait(&barrier);
    }
    
    pthread_exit(NULL);
}

void freeGraph(Graph* g) {
    for (int i = 0; i < g->V; i++) free(g->adjMatrix[i]);
    free(g->adjMatrix);
    free(g);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <graph_file_number>\n", argv[0]);
        return 1;
    }

    char filename[50];
    sprintf(filename, "graphs/graph_%d.txt", atoi(argv[1]));
    Graph* g = readGraph(filename);
    if (!g) return 1;

    int num_threads = 8;
    bool debug = true;
    int V = g->V;

    // Allocate memory
    int *d = malloc(V * sizeof(int));
    int *d_local = malloc(V * sizeof(int));
    int *p = malloc(V * sizeof(int));
    int *mask = malloc(V * sizeof(int));
    int *mask1 = malloc(V * sizeof(int));
    int flag = 1;
	
	// Time only the parallel computation
    long long tstart = gettime();

    // Initialize threads for parallel initialization
    pthread_t init_threads[num_threads];
    ThreadData init_data[num_threads];
    for (int i = 0; i < num_threads; i++) {
        init_data[i] = (ThreadData){
            .graph = g, .d = d, .d_local = d_local,
            .mask = mask, .mask1 = mask1, .p = p,
            .flag = &flag, .thread_id = i, .num_threads = num_threads
        };
        pthread_create(&init_threads[i], NULL, parallel_init, &init_data[i]);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(init_threads[i], NULL);
    }

    // Initialize synchronization primitives
    pthread_mutex_init(&mutex, NULL);
    pthread_barrier_init(&barrier, NULL, num_threads);


    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    for (int i = 0; i < num_threads; i++) {
        thread_data[i] = (ThreadData){
            .graph = g, .d = d, .d_local = d_local,
            .mask = mask, .mask1 = mask1, .p = p,
            .flag = &flag, .thread_id = i, .num_threads = num_threads
        };
        pthread_create(&threads[i], NULL, bellmanford_parallel_helper, &thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    long long tstop = gettime();

    // Cleanup
    pthread_mutex_destroy(&mutex);
    pthread_barrier_destroy(&barrier);

    if (debug) print_results(d, p, V);

    printf("Parallel computation time: %.6f seconds\n", (double)(tstop - tstart) / 1e9);

    free(d); free(d_local); free(p); free(mask); free(mask1);
    freeGraph(g);
    return 0;
}