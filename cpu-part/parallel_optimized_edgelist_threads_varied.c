#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <limits.h>
#include <stdbool.h>

#define INFINITY INT_MAX

// Cross-platform timing
#ifdef _WIN32
#include <windows.h>
long long get_nanoseconds() {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (counter.QuadPart * 1000000000LL) / freq.QuadPart;
}
#else
#include <time.h>
long long get_nanoseconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
#endif

typedef struct Edge {
    int u, v, w;
} Edge;

typedef struct Graph {
    int V, E;
    Edge *edge;
} Graph;

typedef struct {
    Graph *g;
    int *d;
    int *p;
    int start, end;
    pthread_mutex_t *mutex;
} ThreadArgs;

typedef struct {
    int *d;
    int *p;
    pthread_mutex_t *mutex;
    int thread_id;
    int num_threads;
    int V;
} InitThreadData;

Graph* createGraph(int V, int E) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->V = V;
    graph->E = E;
    graph->edge = (Edge*)malloc(E * sizeof(Edge));
    return graph;
}

Graph* readGraph(const char* filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return NULL;
    }

    int V, E;
    fscanf(file, "%d\n%d\n", &E, &V);
    printf("Graph: %d edges, %d vertices\n", E, V);

    Graph* graph = createGraph(V, E);
    char line[2048];
    int i = 0;

    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ":");
        int u = atoi(token);

        while ((token = strtok(NULL, ";"))) {
            int v, w;
            sscanf(token, "%d,%d", &v, &w);
            if (i == 0 || !(v == graph->edge[i-1].v && u == graph->edge[i-1].u)) {
                graph->edge[i++] = (Edge){u, v, w};
            }
        }
    }
    fclose(file);
    return graph;
}

void* parallel_init(void* arg) {
    InitThreadData* data = (InitThreadData*)arg;
    int first = (data->thread_id * data->V) / data->num_threads;
    int last = ((data->thread_id + 1) * data->V) / data->num_threads;

    for (int i = first; i < last; i++) {
        data->d[i] = INFINITY;
        data->p[i] = -1;
        pthread_mutex_init(&data->mutex[i], NULL);
    }

    if (data->thread_id == 0) {
        data->d[0] = 0;
    }

    pthread_exit(NULL);
}

void* relax_edges(void *args) {
    ThreadArgs *targs = (ThreadArgs*)args;
    for (int j = targs->start; j < targs->end; j++) {
        int u = targs->g->edge[j].u;
        int v = targs->g->edge[j].v;
        int w = targs->g->edge[j].w;
        
        if (targs->d[u] != INFINITY && targs->d[v] > targs->d[u] + w) {
            pthread_mutex_lock(&targs->mutex[v]);
            if (targs->d[v] > targs->d[u] + w) {
                targs->d[v] = targs->d[u] + w;
                targs->p[v] = u;
            }
            pthread_mutex_unlock(&targs->mutex[v]);
        }
    }
    return NULL;
}

bool detect_negative_cycles(Graph *g, const int *d, const int *p) {
    for (int i = 0; i < g->E; i++) {
        int u = g->edge[i].u;
        int v = g->edge[i].v;
        int w = g->edge[i].w;
        if (d[u] != INFINITY && d[v] > d[u] + w) {
            return true;
        }
    }
    return false;
}

void print_results(const int *d, const int *p, int V) {
    printf("Distance array: ");
    for (int i = 0; i < V; i++) {
        printf(d[i] == INFINITY ? "INF " : "%d ", d[i]);
    }
    printf("\nPredecessor array: ");
    for (int i = 0; i < V; i++) {
        printf("%d ", p[i]);
    }
    printf("\n");
}

void print_usage(const char *program_name) {
    fprintf(stderr, "Usage: %s <graph_file_number> <num_threads> [debug]\n", program_name);
    fprintf(stderr, "  num_threads: Number of threads to use (1-%d)\n", 256);
    fprintf(stderr, "  debug: 0 (silent) or 1 (verbose, default)\n");
}

int main(int argc, char *argv[]) {
    if (argc < 3 || argc > 4) {
        print_usage(argv[0]);
        return 1;
    }

    int graph_num = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    bool debug = (argc == 4) ? (atoi(argv[3]) != 0) : true;

    // Validate thread count
    if (num_threads < 1 || num_threads > 256) {
        fprintf(stderr, "Error: Number of threads must be between 1 and 256\n");
        print_usage(argv[0]);
        return 1;
    }

    char filename[50];
    sprintf(filename, "graphs/graph_%d.txt", graph_num);
    Graph* g = readGraph(filename);
    if (!g) return 1;

    int *d = (int*)malloc(g->V * sizeof(int));
    int *p = (int*)malloc(g->V * sizeof(int)); 
    pthread_mutex_t mutex[g->V];

    long long tstart = get_nanoseconds();

    // Parallel initialization
    pthread_t init_threads[num_threads];
    InitThreadData init_data[num_threads];
    for (int i = 0; i < num_threads; i++) {
        init_data[i] = (InitThreadData){
            .d = d, .p = p, .mutex = mutex,
            .thread_id = i, .num_threads = num_threads, .V = g->V
        };
        pthread_create(&init_threads[i], NULL, parallel_init, &init_data[i]);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(init_threads[i], NULL);
    }

    // Parallel relaxation
    pthread_t threads[num_threads];
    ThreadArgs targs[num_threads];
    int chunk_size = g->E / num_threads;

    for (int i = 1; i < g->V; i++) {
        for (int t = 0; t < num_threads; t++) {
            targs[t] = (ThreadArgs){
                .g = g, .d = d, .p = p,
                .start = t * chunk_size,
                .end = (t == num_threads - 1) ? g->E : (t + 1) * chunk_size,
                .mutex = mutex
            };
            pthread_create(&threads[t], NULL, relax_edges, &targs[t]);
        }
        for (int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], NULL);
        }
    }

    // Negative cycle detection
    if (detect_negative_cycles(g, d, p)) {
        printf("Negative weight cycle detected!\n");
    }

    long long tstop = get_nanoseconds();

    if (debug) {
        print_results(d, p, g->V);
    }

    printf("Algorithm time: %.6f milliseconds (using %d threads)\n", 
           (double)(tstop - tstart) / 1e6, num_threads);

    // Cleanup
    for (int i = 0; i < g->V; i++) {
        pthread_mutex_destroy(&mutex[i]);
    }
    free(d);
    free(p);
    free(g->edge);
    free(g);
    return 0;
}