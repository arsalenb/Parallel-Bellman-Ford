#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <limits.h>

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
double gettime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1e9) + ts.tv_nsec;
}
#endif

// Graph structure using adjacency matrix
typedef struct Graph {
    int V;             // Number of vertices
    int E;             // Number of edges
    int** adjMatrix;   // Adjacency matrix
} Graph;

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

// Create a graph with V vertices
Graph* createGraph(int V) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->V = V;
    graph->E = 0;
    
    graph->adjMatrix = (int**)malloc(V * sizeof(int*));
    for (int i = 0; i < V; i++) {
        graph->adjMatrix[i] = (int*)malloc(V * sizeof(int));
        for (int j = 0; j < V; j++) {
            graph->adjMatrix[i][j] = 0;
        }
    }
    
    return graph;
}

void addEdge(Graph* graph, int u, int v, int w) {
    if (graph->adjMatrix[u][v] == 0) {
        graph->E++;
    }
    graph->adjMatrix[u][v] = w;
}

Graph* readGraph(char* filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Could not open file for reading\n");
        return NULL;
    }

    int V, E;
    fscanf(file, "%d\n", &E);
    fscanf(file, "%d\n", &V);

    Graph* graph = createGraph(V);
    char line[2048];
    char* token;

    while(fgets(line, sizeof(line), file) != NULL) {
        token = strtok(line, ":");
        int u = atoi(token);

        while((token = strtok(NULL, ";")) != NULL) {
            int v, w;
            sscanf(token, "%d,%d", &v, &w);
            addEdge(graph, u, v, w);
        }
    }
    fclose(file);
    return graph;
}

void* bellmanford_parallel_helper(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    Graph* g = data->graph;
    int V = g->V;
    
    int first = (data->thread_id * V) / data->num_threads;
    int last = ((data->thread_id + 1) * V) / data->num_threads;
    
    for (int k = 0; k < V; k++) {
        // Relaxation phase
        for (int u = first; u < last; u++) {
            for (int v = 0; v < V; v++) {
                int w = g->adjMatrix[u][v];
                if (w != 0 && data->mask[u] && data->d[u] != INFINITY && 
                    data->d_local[v] > data->d[u] + w) {
                    data->d_local[v] = data->d[u] + w;
                    data->mask1[v] = 1;
                    data->p[v] = u;
                }
            }
        }
        
        pthread_barrier_wait(&barrier);
        
        // Update global distances
        for (int v = first; v < last; v++) {
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
        if (data->thread_id == 0) {
            *(data->flag) = 0;
        }
        
        for (int v = first; v < last; v++) {
            data->mask[v] = data->mask1[v];
            data->mask1[v] = 0;
        }
        
        pthread_barrier_wait(&barrier);
    }
    
    pthread_exit(NULL);
}

void bellmanford_parallel(Graph* g, int source, int num_threads) {
    int V = g->V;
    
    // Allocate data structures
    int* d = (int*)malloc(V * sizeof(int));
    int* d_local = (int*)malloc(V * sizeof(int));
    int* p = (int*)malloc(V * sizeof(int));
    int* mask = (int*)malloc(V * sizeof(int));
    int* mask1 = (int*)malloc(V * sizeof(int));
    int flag = 1;
    
    // Initialize
    for (int i = 0; i < V; i++) {
        d[i] = INFINITY;
        d_local[i] = INFINITY;
        p[i] = -1;
        mask[i] = 0;
        mask1[i] = 0;
    }
    d[source] = 0;
    d_local[source] = 0;
    mask[source] = 1;
    
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
    
    // Clean up
    pthread_mutex_destroy(&mutex);
    pthread_barrier_destroy(&barrier);
    
    free(d);
    free(d_local);
    free(p);
    free(mask);
    free(mask1);
}

void freeGraph(Graph* g) {
    for (int i = 0; i < g->V; i++) {
        free(g->adjMatrix[i]);
    }
    free(g->adjMatrix);
    free(g);
}

int main() {
	int num_threads = 8;

    FILE* csv = fopen("results.csv", "w");
    if (!csv) {
        printf("Failed to create CSV file.\n");
        return 1;
    }
    fprintf(csv, "Vertices,Edges,Time(seconds)\n");
	
    for (int vertices = 10; vertices <= 2500; vertices += 10) {
        char filename[50];
        sprintf(filename, "graphs/graph_%d.txt", vertices);

        Graph* g = readGraph(filename);
        if (!g) continue;

        // Timed run
        long long tstart = gettime();
		bellmanford_parallel(g, 0, num_threads);
        long long tstop = gettime();

        fprintf(csv, "%d,%d,%.9f\n", vertices, g->E, (double)(tstop - tstart) / 1e9);
        printf("Processed %d vertices: %.9f sec\n", vertices, (double)(tstop - tstart) / 1e9);

        free(g);
    }

    fclose(csv);
    return 0;
}