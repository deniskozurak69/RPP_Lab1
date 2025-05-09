#include <iostream>
#include <vector>
#include <omp.h>
#include <climits>
using namespace std;
const int INF = 1e9;
void initializeGraph(std::vector<std::vector<int>>& dist, int N) 
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (i == j) dist[i][j] = 0;
            else dist[i][j] = rand() % 100 + 1;
        }
    }
}
void floydWarshall(std::vector<std::vector<int>>& dist, int N, int num_threads) 
{
    omp_set_num_threads(num_threads);
    for (int k = 0; k < N; ++k) 
    {
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) 
        {
            for (int j = 0; j < N; ++j) 
            {
                if (dist[i][k] < 1e9 && dist[k][j] < 1e9 && dist[i][k] + dist[k][j] < dist[i][j]) dist[i][j] = dist[i][k] + dist[k][j];
            }
        }
    }
}
int main() 
{
    int N;
    int num_threads;
    cout << "Enter the number of vertexes: ";
    cin >> N;
    cout << "Enter the number of threads: ";
    cin >> num_threads;
    vector<vector<int>> dist(N, vector<int>(N));
    initializeGraph(dist, N);
    double start_time = omp_get_wtime();
    floydWarshall(dist, N, num_threads);
    double end_time = omp_get_wtime();
    std::cout << "Execution time: " << (end_time - start_time) << " seconds" << "\n";
    return 0;
}




