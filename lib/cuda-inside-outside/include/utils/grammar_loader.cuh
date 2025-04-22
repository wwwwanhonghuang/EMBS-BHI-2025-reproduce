#ifndef GRAMMAR_LOADER_CUH
#define GRAMMAR_LOADER_CUH
#include <cuda_runtime.h>
#include "macros.def"
#include <grammar/grammar_parser.hpp>

__host_pt__ float* initialize_grammar_buffer_from_pcfg(pcfg* pcfg_data);
#endif