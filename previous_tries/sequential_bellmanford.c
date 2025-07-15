#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

// Graph structures
typedef struct Edge {
    int u, v, w;  // Edge from u to v with weight w
} Edge;

typedef struct Graph {
    int V, E;      // Vertex and edge counts
    Edge *edge;    // Array of edges
} Graph;

// ------------------------ Graph Creation -------------------------- //
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
            // Skip duplicate edges
            if (i == 0 || !(v == graph->edge[i-1].v && u == graph->edge[i-1].u)) {
                graph->edge[i++] = (Edge){u, v, w};
            }
        }
    }
    fclose(file);
    return graph;
}

// ------------------------ Algorithm Functions -------------------------- //
void initialize_arrays(int *d, int *p, int V, int source) {
    for (int i = 0; i < V; i++) {
        d[i] = INFINITY;
        p[i] = -1;
    }
    d[source] = 0;  // Distance to source is 0
}

void relax_edges(Graph *g, int *d, int *p) {
    for (int i = 0; i < g->E; i++) {
        int u = g->edge[i].u;
        int v = g->edge[i].v;
        int w = g->edge[i].w;
        if (d[u] != INFINITY && d[v] > d[u] + w) {
            d[v] = d[u] + w;
            p[v] = u;
        }
    }
}

bool detect_negative_cycles(Graph *g, const int *d) {
    for (int i = 0; i < g->E; i++) {
        int u = g->edge[i].u;
        int v = g->edge[i].v;
        int w = g->edge[i].w;
        if (d[u] != INFINITY && d[v] > d[u] + w) {
            return true;  // Negative cycle detected
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
    for (int i = 0; i < V; i++) printf("%d ", p[i]);
    printf("\n");
}

// ------------------------ Main -------------------------- //
int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <graph_file_number>\n", argv[0]);
        return 1;
    }

    char filename[50];
	int source = 0;
    sprintf(filename, "graphs/graph_%d.txt", atoi(argv[1]));
    Graph* g = readGraph(filename);
    if (!g) return 1;

    bool debug = true;  // Set to false to disable printing

	// ------------------------ Core Algorithm -------------------------- //

    int V = g->V;
    int *d = (int*)malloc(V * sizeof(int));
    int *p = (int*)malloc(V * sizeof(int));
	
	// Time only the algorithm (excludes graph loading)
	long long tstart = get_nanoseconds();
    // Step 1: Initialize
    initialize_arrays(d, p, V, source);

    // Step 2: Relax edges V-1 times
    for (int i = 1; i <= V - 1; i++) {
        relax_edges(g, d, p);
    }

    // Step 3: Check for negative cycles
    if (detect_negative_cycles(g, d)) {
        printf("Negative weight cycle detected!\n");
    } 
	
	long long tstop = get_nanoseconds();

	if (debug) {
        print_results(d, p, V);
    }

    free(d);
    free(p);

    printf("Algorithm time: %.6f seconds\n", (double)(tstop - tstart) / 1e9);

    free(g->edge);
    free(g);
    return 0;
}