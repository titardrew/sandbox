#include <cassert>

#include "sse_v0.h"
#include "sse_v1.h"
#include "scalar.h"
#include "scalar_unrolled.h"

void ClogMemory()
{
    constexpr size_t kSize = 64*1024*1024;
    volatile static int32_t *arr = new int32_t[kSize];

    for (size_t j = 0; j < kSize; ++j) {
        arr[j] = rand();
    }
}

#define RUN_EXPERIMENT(tri_count_func__, edges__)    \
    do {                                             \
        ClogMemory();                                \
        printf("\nRunning %s\n", #tri_count_func__); \
        Profile p__;                                 \
        int64_t num_tri = tri_count_func__(edges__); \
        printf("Num triangles: %zd\n", num_tri);     \
    } while(0)

void Run(const AlignedIntArray &edges)
{
    printf("Total num edges: %zu\n", edges.size());
    RUN_EXPERIMENT(CountTri_N2_LinearSearchSSE, edges);
    RUN_EXPERIMENT(CountTri_N2, edges);
    RUN_EXPERIMENT(CountTri_N2_LinearSearch_Unrolled2, edges);
    RUN_EXPERIMENT(CountTri_N2_LinearSearch_Unrolled1, edges);
    RUN_EXPERIMENT(CountTri_SSE4, edges);
    RUN_EXPERIMENT(CountTri_SSE1, edges);
    RUN_EXPERIMENT(CountTri_SSE_Branchless, edges);
    RUN_EXPERIMENT(CountTri_ScalarUR, edges);
    // RUN_EXPERIMENT(CountTri_N2_BinarySearch, edges);
    // RUN_EXPERIMENT(CountTri_NaiveN3, edges);
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Usage: %s /path/to/data.txt\n", argv[0]);
        return 1;
    }
    char *path = argv[1];
    FILE *f_example = fopen(path, "rt");
    if (!f_example) {
        printf("No such file: %s\n", path);
        return 1;
    }

    std::vector<int> in_edges;
    in_edges.reserve(50);
    while (!feof(f_example)) {
        int number = -1;
        int n_read = fscanf(f_example, "%d", &number);
        assert(n_read == 1);
        in_edges.push_back(number);
    }
    fclose(f_example);

    AlignedIntArray edges(in_edges);
    in_edges.clear();

    // asm("int $3");
    Run(edges);
    return 0;
}
