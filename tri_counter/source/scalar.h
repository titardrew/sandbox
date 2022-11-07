#pragma once

#include <cstdint>
#include <cstdio>

#include "helpers.h"

int64_t CountTri_NaiveN3(AlignedIntArray edges);
int64_t CountTri_N2(AlignedIntArray edges);
int64_t CountTri_N2_BinarySearch(AlignedIntArray edges);
