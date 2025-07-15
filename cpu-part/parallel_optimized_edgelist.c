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
    return (ts.tv_sec * 1e9) + ts.tv_nsec;
}
#endif

typedef struct Edge { int u, v, w; } Edge;
typedef struct Graph { int V, E; Edge *edge; } Graph;

// Parallel computation data
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

// Create a graph with V vertices and E edges
Graph* createGraph(int V, int E) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->V = V;
    graph->E = E;
    graph->edge = (Edge*)malloc(E * sizeof(Edge));
    return graph;
}

Graph* readGraph(char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Could not open file: %s\n", filename);
        return NULL;
    }

    int V, E;
    fscanf(file, "%d\n%d\n", &E, &V);

    Graph* graph = createGraph(V, E);
    char line[2048];
    int i = 0;

    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ":");
        int u = atoi(token);

        while ((token = strtok(NULL, ";"))) {
            int v, w;
            sscanf(token, "%d,%d", &v, &w);
            graph->edge[i].u = u;
            graph->edge[i].v = v;
            graph->edge[i].w = w;
            i++;
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
    int E = g->E;
    int tid = data->thread_id;
    int num_threads = data->num_threads;

    // Calculate edge range for this thread
    int first_edge = (tid * E) / num_threads;
    int last_edge = ((tid + 1) * E) / num_threads;

    // Calculate vertex range for updates and mask reset
    int first_vertex = (tid * V) / num_threads;
    int last_vertex = ((tid + 1) * V) / num_threads;

    for (int k = 0; k < V; k++) {
        // Relaxation phase: process assigned edges
        for (int i = first_edge; i < last_edge; i++) {
            Edge e = g->edge[i];
            int u = e.u;
            int v = e.v;
            int w = e.w;

            if (data->mask[u] && data->d[u] != INFINITY && data->d_local[v] > data->d[u] + w) {
                // Atomic update to d_local[v]
                int new_dist = data->d[u] + w;
                if (new_dist < data->d_local[v]) {
                    pthread_mutex_lock(&mutex);
                    if (new_dist < data->d_local[v]) {
                        data->d_local[v] = new_dist;
                        data->mask1[v] = 1;
                        data->p[v] = u;
                    }
                    pthread_mutex_unlock(&mutex);
                }
            }
        }

        pthread_barrier_wait(&barrier);

        // Update global distances for assigned vertices
        for (int v = first_vertex; v < last_vertex; v++) {
            if (data->d_local[v] != data->d[v]) {
                data->d[v] = data->d_local[v];
                pthread_mutex_lock(&mutex);
                *(data->flag) = 1;
                pthread_mutex_unlock(&mutex);
            }
        }

        pthread_barrier_wait(&barrier);

        // Check for early termination
        if (*(data->flag) == 0) {
            pthread_barrier_wait(&barrier);
            break;
        }

        pthread_barrier_wait(&barrier);

        // Reset flag and update masks
        if (tid == 0) {
            *(data->flag) = 0;
        }

        // Update masks for assigned vertices
        for (int v = first_vertex; v < last_vertex; v++) {
            data->mask[v] = data->mask1[v];
            data->mask1[v] = 0;
        }

        pthread_barrier_wait(&barrier);
    }

    pthread_exit(NULL);
}

void freeGraph(Graph* g) {
    free(g->edge);
    free(g);
}

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <graph_file_number> [debug]\n", argv[0]);
        fprintf(stderr, "  debug: 0 (silent) or 1 (verbose, default)\n");
        return 1;
    }
	
	// Set debug mode (default to true if not specified)
    bool debug = true;
    if (argc == 3) {
        debug = (atoi(argv[2]) != 0);
    }
    
    char filename[50];
    int arg = atoi(argv[1]);
    int num_threads = 8;
    sprintf(filename, "graphs/graph_%d.txt", arg);
    
    Graph* g = readGraph(filename);
    if (!g) return 1;
    
    int V = g->V;
    
    // Allocate data structures
    int* d = (int*)malloc(V * sizeof(int));
    int* d_local = (int*)malloc(V * sizeof(int));
    int* p = (int*)malloc(V * sizeof(int));
    int* mask = (int*)malloc(V * sizeof(int));
    int* mask1 = (int*)malloc(V * sizeof(int));
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
    
    // Create threads
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].graph = g;
        thread_data[i].d = d;
        thread_data[i].d_local = d_local;
        thread_data[i].mask = mask;
        thread_data[i].mask1 = mask1;
        thread_data[i].p = p;
        thread_data[i].flag = &flag;
        thread_data[i].thread_id = i;
        thread_data[i].num_threads = num_threads;
        
        pthread_create(&threads[i], NULL, bellmanford_parallel_helper, &thread_data[i]);
    }
    
    // Wait for threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    long long tstop = gettime();

    // Cleanup
    pthread_mutex_destroy(&mutex);
    pthread_barrier_destroy(&barrier);
    
    // Print results
    if (debug) print_results(d, p, V);

    printf("Algorithm time: %.6f milliseconds\n", (double)(tstop - tstart) / 1e6);

    
    free(d); free(d_local); free(p); free(mask); free(mask1);

    freeGraph(g);
    return 0;
}