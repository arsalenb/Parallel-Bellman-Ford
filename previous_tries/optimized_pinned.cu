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

    if (mask[u] == 1 && d[u] != INFINITY) {
        int new_dist = d[u] + w;
        int old_dist = atomicMin(&d[v], new_dist);
        if (old_dist > new_dist) {
            p[v] = u;
            mask1[v] = 1;
            *flag = 1;
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

void cuda_bellmanford(Graph* g, int source) {
    Edge* d_edges;
    int *d_d, *d_p, *d_mask, *d_mask1, *d_flag;
    int *h_d, *h_p;

    // Allocate host memory
    h_d = (int*)malloc(g->V * sizeof(int));
    h_p = (int*)malloc(g->V * sizeof(int));

    // Allocate device memory
    cudaMalloc(&d_edges, g->E * sizeof(Edge));
    cudaMalloc(&d_d, g->V * sizeof(int));
    cudaMalloc(&d_p, g->V * sizeof(int));
    cudaMalloc(&d_mask, g->V * sizeof(int));
    cudaMalloc(&d_mask1, g->V * sizeof(int));
    cudaMalloc(&d_flag, sizeof(int));

    // Create two CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Split work between two streams
    int split_point = g->E / 2;
    
    // Copy graph data to device (async in stream1)
    cudaMemcpyAsync(d_edges, g->edge, g->E * sizeof(Edge), cudaMemcpyHostToDevice, stream1);

    // Initialize device arrays (async in stream2)
    int threadsPerBlock = 1024;
    int blocks = (g->V + threadsPerBlock - 1) / threadsPerBlock;
    init_kernel<<<blocks, threadsPerBlock, 0, stream2>>>(d_d, d_p, d_mask, d_mask1, g->V, source);

    // Synchronize before starting main loop
    cudaDeviceSynchronize();

    // Run relaxation until no more updates or V-1 iterations
    int relaxBlocks1 = (split_point + threadsPerBlock - 1) / threadsPerBlock;
    int relaxBlocks2 = (g->E - split_point + threadsPerBlock - 1) / threadsPerBlock;
    int h_flag;
    
    for (int i = 0; i < g->V - 1; i++) {
        h_flag = 0;
        cudaMemcpyAsync(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice, stream1);
        
        // Launch kernels in both streams
        relax_kernel<<<relaxBlocks1, threadsPerBlock, 0, stream1>>>(
            d_edges, d_d, d_p, d_mask, d_mask1, d_flag, split_point);
        
        relax_kernel<<<relaxBlocks2, threadsPerBlock, 0, stream2>>>(
            d_edges + split_point, d_d, d_p, d_mask, d_mask1, d_flag, g->E - split_point);
        
        cudaMemcpyAsync(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost, stream1);
        cudaStreamSynchronize(stream1);
        
        if (h_flag == 0) {
            break;
        }
        
        swap_kernel<<<blocks, threadsPerBlock, 0, stream1>>>(d_mask, d_mask1, d_flag, g->V);
        cudaStreamSynchronize(stream1);
    }

    // Copy results back (async in stream1)
    cudaMemcpyAsync(h_d, d_d, g->V * sizeof(int), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(h_p, d_p, g->V * sizeof(int), cudaMemcpyDeviceToHost, stream2);
    cudaDeviceSynchronize();

    printf("Distance array: ");
    display(h_d, g->V);
    printf("Predecessor array: ");
    display(h_p, g->V);

    // Cleanup
    free(h_d);
    free(h_p);
    cudaFree(d_edges);
    cudaFree(d_d);
    cudaFree(d_p);
    cudaFree(d_mask);
    cudaFree(d_mask1);
    cudaFree(d_flag);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <number of vertices>\n", argv[0]);
        return 1;
    }

    char filename[50];
    sprintf(filename, "graphs/graph_%d.txt", atoi(argv[1]));
    Graph* g = readGraph(filename);
    if (!g) return 1;

    double tstart, tstop;

    tstart = gettime();
    cuda_bellmanford(g, 0);
    tstop = gettime();
    printf("Total algorithm time: %.9f seconds\n", tstop - tstart);

    free(g->edge);
    free(g);

    return 0;
}