#include "scalar.h"

#include <algorithm>

int64_t CountTri_NaiveN3(AlignedIntArray edges)
{
    std::sort(edges.begin(), edges.end());
    size_t N = edges.size();
    int64_t num_tri = 0;
    for (size_t i = 0; i < N - 2; ++i) {
        for (size_t j = i + 1; j < N - 1; ++j) {
            for (size_t k = j + 1; k < N; ++k) {
                if (edges[i] + edges[j] > edges[k]) {
                    ++num_tri;
                }
            }
        }
    }
    return num_tri;
}

int64_t CountTri_N2(AlignedIntArray edges)
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

        // this test saves us about 12% on larger datasets.
        if (edges[i_upper - 1] + edges[i_upper] <= edges[i_largest]) continue;

        while (i_lower < i_upper) {
            if (edges[i_lower] + edges[i_upper] > edges[i_largest]) {
#ifdef USE_OPENMP
                #pragma omp atomic
#endif
                num_tri += i_upper - i_lower;
                --i_upper;
            } else {
                ++i_lower;
            }
        }
    }
    return num_tri;
}

static size_t BinarySearch(const AlignedIntArray &arr, ssize_t i_lower, ssize_t i_upper, int ref)
{
    size_t result = i_upper;

    i_upper -= 1;
    while (i_upper >= i_lower) {
        size_t i_middle = (i_upper + i_lower) / 2;
        if (arr[i_middle] > ref) {
            i_upper = i_middle - 1;
            result = i_middle;
        } else {
            i_lower = i_middle + 1;
        }
    }
    return result;
}

/*
static size_t BinarySearch_Branchless(const AlignedIntArray &arr, ssize_t i_lower, ssize_t i_upper, int ref)
{
    while (i_lower < i_upper) {
        ssize_t i_middle = (i_lower + i_upper) >> 1;
        asm("cmpl %3, %2\n\tcmovg %4, %0\n\tcmovle %5, %1"
             : "+r" (i_lower),
               "+r" (i_upper)
             : "r" (ref), "g" (arr[i_middle]),
               "g" (i_middle + 1), "g" (i_middle));
    }
    return i_upper;
}
*/

// Must be equivalent to N2.
int64_t CountTri_N2_BinarySearch(AlignedIntArray edges)
{
    std::sort(edges.begin(), edges.end());
    size_t N = edges.size();
    int64_t num_tri = 0;
    size_t bin_counter = 0, lin_counter = 0;
    for (size_t i_largest = 2; i_largest < N; ++i_largest) {
        // printf("iter = %zu (out of %zu)\n", i_largest, N);

        ssize_t i_lower = 0;
        ssize_t i_upper = i_largest - 1;

        if (edges[i_upper - 1] + edges[i_upper] <= edges[i_largest]) continue;

        while (i_lower < i_upper) {
            auto i_lower_before = i_lower;
            i_lower = BinarySearch(edges, i_lower, i_upper, edges[i_largest] - edges[i_upper]);
            // fprintf(stderr, "%zd, %zd, %zu, %zd, %d\n", i_lower, i_upper, i_largest, i_lower - i_lower_before, edges[i_largest] - edges[i_upper]);
            num_tri += i_upper - i_lower;
            --i_upper;
        }
    }
    return num_tri;
}
