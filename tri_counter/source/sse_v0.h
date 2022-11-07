#pragma once

#include <cstdio>

#include "helpers.h"

size_t CountTri_SSE4(const AlignedIntArray &edges);
size_t CountTri_SSE1(const AlignedIntArray &edges);
size_t CountTri_SSE_Branchless(const AlignedIntArray &edges);
size_t CountTri_ScalarUR(const AlignedIntArray &edges);
