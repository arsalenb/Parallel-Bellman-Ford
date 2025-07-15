#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INFINITY 99999

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

// Node structure for adjacency list
typedef struct Node {
    int vertex;         // Destination vertex
    int weight;         // Edge weight
    struct Node* next;  // Pointer to next node
} Node;

// Graph structure using adjacency list
typedef struct Graph {
    int V;             // Number of vertices
    int E;             // Number of edges
    Node** adjList;    // Array of adjacency lists
} Graph;

// Create a new node for adjacency list
Node* createNode(int v, int w) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->vertex = v;
    newNode->weight = w;
    newNode->next = NULL;
    return newNode;
}

// Create a graph with V vertices
Graph* createGraph(int V) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->V = V;
    graph->E = 0;
    graph->adjList = (Node**)malloc(V * sizeof(Node*));
    
    for (int i = 0; i < V; i++) {
        graph->adjList[i] = NULL;
    }
    
    return graph;
}

// Add an edge to the graph
void addEdge(Graph* graph, int u, int v, int w) {
    // Add edge from u to v
    Node* newNode = createNode(v, w);
    newNode->next = graph->adjList[u];
    graph->adjList[u] = newNode;
    graph->E++;
}

// Read graph from file and build adjacency list
Graph* readGraph(char* filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Could not open file for reading\n");
        return NULL;
    }

    int V, E;
    fscanf(file, "%d\n", &E);
    fscanf(file, "%d\n", &V);
    printf("Edges: %d, Vertices: %d\n", E, V);

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

// Bellman-Ford algorithm using adjacency list
void bellmanford(Graph *g, int source) {
    int tV = g->V;  // Total vertices

    int *d = (int*)malloc(tV * sizeof(int));
    int *p = (int*)malloc(tV * sizeof(int));

    // Step 1: Initialize distance and predecessor arrays
    for (int i = 0; i < tV; i++) {
        d[i] = INFINITY;
        p[i] = -1;
    }
    d[source] = 0;

    // Step 2: Relax edges |V| - 1 times
    for (int i = 1; i <= tV - 1; i++) {
        for (int u = 0; u < tV; u++) {
            Node* node = g->adjList[u];
            while (node != NULL) {
                int v = node->vertex;
                int w = node->weight;
                
                if (d[u] != INFINITY && d[v] > d[u] + w) {
                    d[v] = d[u] + w;
                    p[v] = u;
                }
                node = node->next;
            }
        }
    }

    // Step 3: Detect negative cycles
    for (int u = 0; u < tV; u++) {
        Node* node = g->adjList[u];
        while (node != NULL) {
            int v = node->vertex;
            int w = node->weight;
            if (d[u] != INFINITY && d[v] > d[u] + w) {
                printf("Negative weight cycle detected!\n");
                free(d);
                free(p);
                return;
            }
            node = node->next;
        }
    }

    // No negative weight cycle found
    printf("Distance array: ");
    for (int i = 0; i < tV; i++) {
        if (d[i] == INFINITY) {
            printf("INF ");
        } else {
            printf("%d ", d[i]);
        }
    }
    printf("\n");

    printf("Predecessor array: ");
    for (int i = 0; i < tV; i++) {
        printf("%d ", p[i]);
    }
    printf("\n");

    free(d);
    free(p);
}

// Free graph memory
void freeGraph(Graph* g) {
    for (int i = 0; i < g->V; i++) {
        Node* node = g->adjList[i];
        while (node != NULL) {
            Node* temp = node;
            node = node->next;
            free(temp);
        }
    }
    free(g->adjList);
    free(g);
}

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

    freeGraph(g);
    return 0;
}