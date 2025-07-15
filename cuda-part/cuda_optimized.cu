#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
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
    return (ts.tv_sec + (double)ts.tv_nsec / 1e9);
}
#endif

typedef struct Edge {
    int u;
    int v;
    int w;
} Edge;

typedef struct Graph {
    int V;
    int E;
    struct Edge *edge;
} Graph;

// CUDA kernels
__global__ void init_kernel(int* d, int* p, int* mask, int* mask1, int V, int source) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < V) {
        d[idx] = (idx == source) ? 0 : INFINITY;
        p[idx] = -1;
        mask[idx] = (idx == source) ? 1 : 0;
        mask1[idx] = 0;
    }
}

__global__ void relax_kernel(Edge* edges, int* d, int* p, int* mask, int* mask1, int* flag, int E) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E) return;

    Edge e = edges[idx];
    int u = e.u;
    int v = e.v;
    int w = e.w;

    if (mask[u] == 1) {
        if (d[u] != INFINITY) {
            int new_dist = d[u] + w;
            int old_dist = atomicMin(&d[v], new_dist);
            if (old_dist > new_dist) {
                p[v] = u;
                mask1[v] = 1;
                *flag = 1;
            }
        }
    }
}

__global__ void swap_kernel(int* mask, int* mask1, int* flag, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < V) {
        mask[idx] = mask1[idx];
        mask1[idx] = 0;
    }
    if (idx == 0) {
        *flag = 0;
    }
}

Graph* createGraph(int V, int E) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->V = V;
    graph->E = E;
    graph->edge = (Edge*)malloc(graph->E * sizeof(Edge));
    return graph;
}

Graph* readGraph(char* filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Could not open file for reading\n");
        return NULL;
    }

    int V, E;
    fscanf(file, "%d\n", &E);
    fscanf(file, "%d\n", &V);
    Graph* graph = createGraph(V, E);
    int i = 0;
    char line[2048];

    while(fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ":");
        int u = atoi(token);

        while((token = strtok(NULL, ";"))) {
            int v, w;
            sscanf(token, "%d,%d", &v, &w);
            if (i > 0 && u == graph->edge[i-1].u && v == graph->edge[i-1].v)
                continue;
            graph->edge[i].u = u;
            graph->edge[i].v = v;
            graph->edge[i].w = w;
            i++;
        }
    }
    fclose(file);
    return graph;
}

void display(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == INFINITY) {
            printf("INF ");
        } else {
            printf("%d ", arr[i]);
        }
    }
    printf("\n");
}

void cuda_bellmanford(Graph* g, int source, bool debug) {
    Edge* d_edges;
    int *d_d, *d_p, *d_mask, *d_mask1, *d_flag;
    int *h_d, *h_p;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate device memory
    cudaMalloc(&d_edges, g->E * sizeof(Edge));
    cudaMemcpy(d_edges, g->edge, g->E * sizeof(Edge), cudaMemcpyHostToDevice);
    cudaMalloc(&d_d, g->V * sizeof(int));
    cudaMalloc(&d_p, g->V * sizeof(int));
    cudaMalloc(&d_mask, g->V * sizeof(int));
    cudaMalloc(&d_mask1, g->V * sizeof(int));
    cudaMalloc(&d_flag, sizeof(int));

    // Record start time
    cudaEventRecord(start, 0);

    // Initialize device arrays
    int threadsPerBlock = 512;
    int blocks = (g->V + threadsPerBlock - 1) / threadsPerBlock;
    init_kernel<<<blocks, threadsPerBlock>>>(d_d, d_p, d_mask, d_mask1, g->V, source);
    cudaDeviceSynchronize();

    // Run relaxation until no more updates or V-1 iterations
    int relaxBlocks = (g->E + threadsPerBlock - 1) / threadsPerBlock;
    int h_flag;
    
    for (int i = 0; i < g->V - 1; i++) {
        h_flag = 0;
        cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice);
        
        relax_kernel<<<relaxBlocks, threadsPerBlock>>>(d_edges, d_d, d_p, d_mask, d_mask1, d_flag, g->E);
        cudaDeviceSynchronize();
        
        cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_flag == 0) {
            break;
        }
        
        swap_kernel<<<blocks, threadsPerBlock>>>(d_mask, d_mask1, d_flag, g->V);
        cudaDeviceSynchronize();
    }

    // Record stop time and synchronize
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float executionTime;
    cudaEventElapsedTime(&executionTime, start, stop);

    // Allocate host memory for results and copy
    h_d = (int*)malloc(g->V * sizeof(int));
    h_p = (int*)malloc(g->V * sizeof(int));
    cudaMemcpy(h_d, d_d, g->V * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_p, d_p, g->V * sizeof(int), cudaMemcpyDeviceToHost);

    if (debug) {
        printf("Distance array: ");
        display(h_d, g->V);
        printf("Predecessor array: ");
        display(h_p, g->V);
    }
    printf("Total algorithm time: %.3f milliseconds\n", executionTime);

    // Cleanup
    free(h_d);
    free(h_p);
    cudaFree(d_edges);
    cudaFree(d_d);
    cudaFree(d_p);
    cudaFree(d_mask);
    cudaFree(d_mask1);
    cudaFree(d_flag);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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
    sprintf(filename, "graphs/graph_%d.txt", atoi(argv[1]));
    Graph* g = readGraph(filename);
    if (!g) return 1;

    cuda_bellmanford(g, 0, debug);

    free(g->edge);
    free(g);

    return 0;
}