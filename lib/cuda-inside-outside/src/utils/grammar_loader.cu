
#include "utils/grammar_loader.cuh"
#include <utils/printer.hpp>

__host_pt__ float* initialize_grammar_buffer_from_pcfg(pcfg* pcfg_data){
    print_map(pcfg_data->nonterminate_map);
    print_map(pcfg_data->reversed_nonterminate_map);
    print_map(pcfg_data->terminate_map);
    print_map(pcfg_data->reversed_terminate_map);

    int NT = pcfg_data->nonterminate_map.size();
    int T = pcfg_data->terminate_map.size();
    int S = T + NT;
    std::cout << "Count of Nonterminate = " << NT << std::endl;
    std::cout << "Count of Terminate = " << T << std::endl;
    std::cout << "Count of Total Symbols = " << S << std::endl;
    std::cout << "Allocate grammar buffer size = (S+1)^3 * sizeof(float) = " 
        << (S + 1) * (S + 1) * (S + 1) << " * " << sizeof(float) << " B = " << (S + 1) * (S + 1) * (S + 1) * sizeof(float) / (1024.0 * 1024.0) << " MB" << std::endl;
    __host_pt__ float* grammar_buffer = new float[(S + 1) * (S + 1) * (S + 1)]();

    std::fill(grammar_buffer, grammar_buffer + (S + 1) * (S + 1) * (S + 1), -INFINITY);

    auto get_symbol_id = [pcfg_data](const std::string& symbol) -> int {
        if(pcfg_data->nonterminate_map.find(symbol) != pcfg_data->nonterminate_map.end()) {
            return 1 + pcfg_data->nonterminate_map[symbol];  // Offset to distinguish nonterminals
        } else if(pcfg_data->terminate_map.find(symbol) != pcfg_data->terminate_map.end()) {
            return 1 + pcfg_data->nonterminate_map.size() + pcfg_data->terminate_map[symbol]; // Offset to distinguish terminals
        } else if(symbol == "") {
            return 0; // Empty string case (possibly representing the epsilon symbol)
        }
        return -1; // If the symbol is not found in either map
    };

    for(auto& lhs_symbol : pcfg_data->grammar_items_map){
        for(auto& grammar_record: pcfg_data->grammar_items_map[lhs_symbol.first]){
            std::cout << "\t" << grammar_record.left << "->" <<
            grammar_record.right1 << " " << 
            grammar_record.right2 << "[" << grammar_record.possibility << " (" << std::exp(grammar_record.possibility) << ")]" <<
            " ===> [" << 
            get_symbol_id(grammar_record.left) << ", " << get_symbol_id(grammar_record.right1) << ", "
                << get_symbol_id(grammar_record.right2) << "]" << std::endl;
            int lhs_id = get_symbol_id(grammar_record.left);
            int rhs_id1 = get_symbol_id(grammar_record.right1);
            int rhs_id2 = get_symbol_id(grammar_record.right2);

            grammar_buffer[lhs_id * (S + 1) * (S + 1) + rhs_id1 * (S + 1) + rhs_id2] = grammar_record.possibility;
        }
    }
    return grammar_buffer;
}
