#include "scalar_unrolled.h"

#include <algorithm>

static size_t LinearSearch1(const AlignedIntArray &arr, size_t i_lower, size_t i_upper, int ref)
{
    {
        ssize_t res = -1;
        #define STEP(i) if (arr[i_lower + i] > ref) res = i_lower + i
        STEP(3);
        STEP(2);
        STEP(1);
        STEP(0);
        #undef STEP

        if (res >= 0) return std::min((size_t)res, i_upper);
    }

    for (size_t i = i_lower; i < i_upper; ++i) {
        if (arr[i] > ref) {
            return i;
        }
    }
    return i_upper;
}

int64_t CountTri_N2_LinearSearch_Unrolled1(AlignedIntArray edges)
{
    std::sort(edges.begin(), edges.end());
    size_t N = edges.size();
    int64_t num_tri = 0;
    for (size_t i_largest = 2; i_largest < N; ++i_largest) {

        size_t i_lower = 0;
        size_t i_upper = i_largest - 1;

        if (edges[i_upper - 1] + edges[i_upper] <= edges[i_largest]) continue;

        while (i_lower < i_upper) {
            i_lower = LinearSearch1(edges, i_lower, i_upper, edges[i_largest] - edges[i_upper]);
            num_tri += i_upper - i_lower;
            --i_upper;
        }
    }
    return num_tri;
}


static size_t LinearSearch2(const AlignedIntArray &arr, size_t i_lower, size_t i_upper, int ref)
{
    {
        size_t res = 0;
        #define STEP(i) res += arr[i_lower + i] <= ref
        STEP(3);
        STEP(2);
        STEP(1);
        STEP(0);
        #undef STEP

        if (res < 4) return std::min(i_upper, res + i_lower);
    }

    for (size_t i = i_lower; i < i_upper; ++i) {
        if (arr[i] > ref) {
            return i;
        }
    }
    return i_upper;
}

int64_t CountTri_N2_LinearSearch_Unrolled2(AlignedIntArray edges)
{
    std::sort(edges.begin(), edges.end());
    size_t N = edges.size();
    int64_t num_tri = 0;

#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (size_t i_largest = 2; i_largest < N; ++i_largest) {

        size_t i_lower = 0;
        size_t i_upper = i_largest - 1;

        if (edges[i_upper - 1] + edges[i_upper] <= edges[i_largest]) continue;

        while (i_lower < i_upper) {
            i_lower = LinearSearch2(edges, i_lower, i_upper, edges[i_largest] - edges[i_upper]);
#ifdef USE_OPENMP
            #pragma omp atomic
#endif
            num_tri += i_upper - i_lower;

            --i_upper;
        }
    }
    return num_tri;
}
