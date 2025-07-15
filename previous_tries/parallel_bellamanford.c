#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>

#define INFINITY 99999

#ifdef _WIN32
#include <windows.h>
long long gettime(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);  // Get clock frequency (Hz)
    QueryPerformanceCounter(&counter); // Get current tick count
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

// Struct for the edges of the graph
typedef struct Edge {
    int u;  // Start vertex of the edge
    int v;  // End vertex of the edge
    int w;  // Weight of the edge (u,v)
} Edge;

// Graph - consists of edges
typedef struct Graph {
    int V;        // Total number of vertices in the graph
    int E;        // Total number of edges in the graph
    struct Edge *edge;  // Array of edges
} Graph;

// ------------------------ CREATE GRAPH -------------------------- //
Graph* createGraph(int V, int E) {
    Graph* graph = (Graph*) malloc(sizeof(Graph));
    graph->V = V;
    graph->E = E;
    graph->edge = (Edge*) malloc(graph->E * sizeof(Edge));

    return graph;
}

Graph* readGraph(char* filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL){
        printf("Could not open file for reading\n");
        return NULL;
    }

    int V, E;
    fscanf(file, "%d\n", &E);
    fscanf(file, "%d\n", &V);
    printf("Edges: %d, Vertices: %d\n", E, V);

    int u, v, w;
    Graph* graph = createGraph(V, E);
    int i = 0;
    char line[2048];
    char* token;

    while(fgets(line, sizeof(line), file) != NULL) {
        token = strtok(line, ":");
        u = atoi(token);

        while((token = strtok(NULL, ";")) != NULL) {
            sscanf(token, "%d,%d", &v, &w);
            if (i != 0 && (v == graph->edge[i-1].v && u == graph->edge[i-1].u)){
                continue;
            }
            graph->edge[i].u = u;
            graph->edge[i].v = v;
            graph->edge[i].w = w;
            i++;
        }
    }
    fclose(file);
    return graph;
}

void bellmanford(struct Graph *g, int source);
void display(int arr[], int size);

// ----------------------- MAIN --------------------------//
int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <number of vertices>\n", argv[0]);
        return 1;
    }
    char filename[50];
    int arg = atoi(argv[1]);
    sprintf(filename, "graphs/graph_%d.txt", arg);
    Graph* g = readGraph(filename);
    if (!g) {
        return 1;
    }

    long long tstart, tstop;

    // Run algorithm
    tstart = gettime();
    bellmanford(g, 0);  // 0 is the source vertex
    tstop = gettime();
	printf("Elapsed time: %.9f seconds\n", (double)(tstop - tstart) / 1e9);

    free(g->edge);
    free(g);
    return 0;
}

// ------------------------- ALGORITHM --------------------//
void bellmanford(struct Graph *g, int source) {
    int *d = (int*)malloc(g->V * sizeof(int));
    for (int i = 0; i < g->V; i++) d[i] = INFINITY;
    d[source] = 0;

    for (int i = 1; i < g->V; i++) {
        #pragma omp parallel for  // Parallelize edge processing
        for (int j = 0; j < g->E; j++) {
            int u = g->edge[j].u, v = g->edge[j].v, w = g->edge[j].w;
            if (d[u] != INFINITY && d[v] > d[u] + w) {
                #pragma omp critical  // Protect concurrent writes
                if (d[v] > d[u] + w) d[v] = d[u] + w;  // Recheck condition
            }
        }
    }
    free(d);
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
