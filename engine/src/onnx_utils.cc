#include "onnx_utils.h"
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <optional>
#include <string>
#include <chrono>

namespace fs = std::filesystem;

std::string findLatestOnnxFile(const std::string& directory) {
    std::string latestFile;
    std::optional<fs::file_time_type> latestTime;

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".onnx") {
            auto ftime = fs::last_write_time(entry);
            if (!latestTime.has_value() || ftime > *latestTime) {
                latestTime = ftime;
                latestFile = entry.path().string();
            }
        }
    }
    return latestFile;
}

std::string getEnginePath(const std::string& onnxPath, const std::string& precision,
                          int batchSize, int deviceId, const std::string& version) {
    fs::path onnx = fs::weakly_canonical(onnxPath);
    std::string modelName = onnx.stem().string();
    std::string directory = onnx.parent_path().string();

    // A TensorRT plan is only valid for the compute capability it was built for
    // and the TensorRT major version that built it. The name used to carry the
    // device *index*, which identifies nothing -- "gpu0" on an RTX 4090 and
    // "gpu0" on an A100 are different plans under the same filename. Anything
    // shipping a prebuilt plan (an image, a shared volume) would silently hand
    // it to hardware that cannot load it.
    //
    // Naming by architecture instead means mismatched plans miss the cache and
    // get rebuilt, rather than colliding.
    std::string hardware = "sm_unknown";
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, deviceId) == cudaSuccess) {
        hardware = "sm" + std::to_string(props.major) + std::to_string(props.minor);
    }

    std::string engineName = modelName + "_" + precision + "_b" + std::to_string(batchSize)
                           + "_" + hardware
                           + "_trt" + std::to_string(NV_TENSORRT_MAJOR)
                           + "_" + version + ".engine";

    return directory.empty() ? engineName : directory + "/" + engineName;
}
