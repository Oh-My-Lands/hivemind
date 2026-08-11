#include "uci.h"
#include "constants.h"
#include "globals.h"
#include "engine.h"
#include "onnx_utils.h"
#include "benchmark.h"
#include "rl/selfplay.h"
#include "rl/model_eval.h"
#include "Fairy-Stockfish/src/bitboard.h"
#include "Fairy-Stockfish/src/position.h"
#include "Fairy-Stockfish/src/thread.h"
#include "Fairy-Stockfish/src/piece.h"
#include "Fairy-Stockfish/src/types.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cstring>

using namespace std;

/**
 * @brief Parses a --*-encoding argument, rejecting anything unrecognised.
 *
 * Deliberately fails the run rather than falling back to a default: a typo'd
 * encoding silently feeds a network the wrong input distribution, and the
 * result of that is a finished match with a believable Elo number in it.
 */
static bool parse_time_encoding(const string& value, TimeEncoding::Mode& out) {
    if (value == "binary") {
        out = TimeEncoding::Mode::BINARY;
        return true;
    }
    if (value == "continuous") {
        out = TimeEncoding::Mode::CONTINUOUS;
        return true;
    }
    cerr << "Error: unknown time encoding '" << value << "' (expected binary or continuous)" << endl;
    return false;
}

static bool parse_alloc_mode(const string& value, TimeAlloc::Mode& out) {
    if (value == "fixed") {
        out = TimeAlloc::Mode::FIXED;
        return true;
    }
    if (value == "flat") {
        out = TimeAlloc::Mode::FLAT;
        return true;
    }
    if (value == "arc") {
        out = TimeAlloc::Mode::ARC;
        return true;
    }
    cerr << "Error: unknown allocation mode '" << value
         << "' (expected fixed, flat or arc)" << endl;
    return false;
}

static const char* alloc_mode_name(TimeAlloc::Mode mode) {
    switch (mode) {
        case TimeAlloc::Mode::FLAT: return "flat";
        case TimeAlloc::Mode::ARC:  return "arc";
        default:                    return "fixed";
    }
}

void printUsage(const char* progName) {
    cout << "Usage: " << progName << " [options]" << endl;
    cout << "Options:" << endl;
    cout << "  --log <level>      Set log level: none, info, debug (default: none)" << endl;
    cout << "  bench [iters]      Run inference benchmark" << endl;
    cout << "  perft [depth]      Run move generation benchmark" << endl;
    cout << "  selfplay [games]   Run RL self-play (default: 1000 games)" << endl;
    cout << "                     One team has time advantage, the other does not" << endl;
    cout << "  eval               Evaluate two models against each other" << endl;
    cout << "    --new <path>     Path to new model ONNX file" << endl;
    cout << "    --old <path>     Path to old model ONNX file" << endl;
    cout << "    --games <n>      Number of games to play (default: 100)" << endl;
    cout << "    --nodes <n>      MCTS nodes per move (default: 800)" << endl;
    cout << "    --time <ms>      Fixed time per move in ms (default: 0, use nodes)" << endl;
    cout << "    --temperature <f> Temperature for opening moves (default: 0.6)" << endl;
    cout << "    --temp-moves <n> Moves before temperature decays to 0 (default: 15)" << endl;
    cout << "    --verbose        Print each game result" << endl;
    cout << "    --pgn <path>     Save games to PGN file" << endl;
    cout << "    --gui            Enable web GUI for live viewing" << endl;
    cout << "    --time-control <ds> Starting clock on all four clocks, deciseconds" << endl;
    cout << "                     (0 = no clock model; 400 is the calibrated working value)" << endl;
    cout << "    --new-encoding <m>  Sit-margin encoding the new net was trained on:" << endl;
    cout << "                     binary or continuous (default: binary)" << endl;
    cout << "    --old-encoding <m>  Same, for the old net (default: binary)" << endl;
    cout << endl;
    cout << "  param-eval         Test same model with different search parameters" << endl;
    cout << "                     --time-control <ds> enables the clock model (0 = off)" << endl;
    cout << "    --model <path>   Path to model ONNX file" << endl;
    cout << "    --games <n>      Number of games to play (default: 100)" << endl;
    cout << "    --encoding <m>   Sit-margin encoding the model was trained on:" << endl;
    cout << "                     binary or continuous (default: binary). One model" << endl;
    cout << "                     plays both sides, so this applies to both." << endl;
    cout << "    --verbose        Print each game result" << endl;
    cout << "    --pgn <path>     Save games to PGN file" << endl;
    cout << "    --gui            Enable web GUI for live viewing" << endl;
    cout << endl;
    cout << "  Player-specific parameters (use --p1-* or --p2-* prefix):" << endl;
    cout << "    --pX-name <s>    Player name (default: 'Player1/2')" << endl;
    cout << "    --pX-nodes <n>   Nodes per move (default: 800)" << endl;
    cout << "    --pX-time <ms>   Time per move in ms (default: 0, use nodes)" << endl;
    cout << "    --pX-batch <n>   Batch size (default: 8)" << endl;
    cout << "    --pX-cpuct <f>   CPUCT init value (default: 2.5)" << endl;
    cout << "    --pX-fpu <f>     FPU reduction (default: 0.4)" << endl;
    cout << "    --pX-contempt <f> Draw contempt (default: 0.12)" << endl;
    cout << "    --pX-pw-coef <f> Progressive widening coefficient (default: 1.0)" << endl;
    cout << "    --pX-pw-exp <f>  Progressive widening exponent (default: 0.3)" << endl;
    cout << "    --pX-mcgs <0|1>  Enable MCGS (default: 1)" << endl;
    cout << "    --pX-tt <0|1>    Enable transpositions (default: 1)" << endl;
    cout << "    --pX-qweight <f> Q-value weight (default: 1.0)" << endl;
    cout << "    --pX-qveto <f>   Q-value veto delta (default: 0.4)" << endl;
    cout << "    --pX-clocks <0|1> Whether this player's search sees the clocks" << endl;
    cout << "                     (default: 1). Set one side to 0 with --time-control" << endl;
    cout << "                     for the Phase 4 A/B." << endl;
}

int main(int argc, char* argv[]) {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " 
                  << cudaGetErrorString(error_id) << std::endl;
        return EXIT_FAILURE;
    }

    // Parse --log argument first (can appear anywhere)
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--help") == 0) || (strcmp(argv[i], "-h") == 0)) {
            printUsage(argv[0]);
            return EXIT_SUCCESS;
        }
        if (strcmp(argv[i], "--log") == 0 && i + 1 < argc) {
            g_logLevel = parseLogLevel(argv[i + 1]);
            // Remove these args from consideration
            for (int j = i; j + 2 < argc; j++) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            i--;  // Recheck this position
        }
    }

    Stockfish::pieceMap.init();
    Stockfish::variants.init();
    Stockfish::Bitboards::init();
    Stockfish::Position::init();
    Stockfish::Threads.set(1);

    init_policy_index();

    // Check for benchmark flag
    if (argc > 1 && string(argv[1]) == "bench") {
        cout << "Running inference benchmark..." << endl;
        Engine engine(0);
        
        const std::string onnxFile = findLatestOnnxFile("./networks");
        if (onnxFile.empty()) {
            cerr << "No ONNX file found in ./networks" << endl;
            return EXIT_FAILURE;
        }
        const std::string engineFile = getEnginePath(onnxFile, "fp16", SearchParams::BATCH_SIZE, 0, "v1");
        
        if (!engine.loadNetwork(onnxFile, engineFile)) {
            cerr << "Failed to load engine" << endl;
            return EXIT_FAILURE;
        }
        
        int iterations = (argc > 2) ? stoi(argv[2]) : 1000;
        benchmark_inference(engine, iterations);
        return EXIT_SUCCESS;
    }

    // Check for perft benchmark flag
    if (argc > 1 && string(argv[1]) == "perft") {
        int depth = (argc > 2) ? stoi(argv[2]) : 5;
        benchmark_movegen(depth);
        return EXIT_SUCCESS;
    }

    // Check for selfplay flag
    if (argc > 1 && string(argv[1]) == "selfplay") {
        cout << "Starting RL self-play..." << endl;
        
        // Parse command line arguments
        RLSettings settings;
        size_t numberOfGames = settings.numberOfGames;
        
        for (int i = 2; i < argc; i++) {
            string arg = argv[i];
            if ((arg == "--games" || arg == "-g") && i + 1 < argc) {
                numberOfGames = stoul(argv[++i]);
            } else if ((arg == "--nodes" || arg == "-n") && i + 1 < argc) {
                settings.nodesPerMove = stoul(argv[++i]);
            }
        }
        
        // Initialize engines for all GPUs
        vector<Engine*> engines;
        for (int i = 0; i < deviceCount; i++) {
            Engine* engine = new Engine(i);
            const std::string onnxFile = findLatestOnnxFile("./networks");
            cout << "Loading model from: " << onnxFile << " on GPU " << i << endl;
            if (onnxFile.empty()) {
                cerr << "No ONNX file found in ./networks" << endl;
                return EXIT_FAILURE;
            }
            const std::string engineFile = getEnginePath(onnxFile, "fp16", SearchParams::BATCH_SIZE, i, "v1");
            cout << "Using TensorRT engine file: " << engineFile << endl;
            if (!engine->loadNetwork(onnxFile, engineFile)) {
                cerr << "Failed to load engine on GPU " << i << endl;
                return EXIT_FAILURE;
            }
            engines.push_back(engine);
        }
        
        run_selfplay(settings, engines, numberOfGames);
        
        // Cleanup
        for (auto* e : engines) {
            delete e;
        }
        
        return EXIT_SUCCESS;
    }

    // Check for eval flag
    if (argc > 1 && string(argv[1]) == "eval") {
        cout << "Starting model evaluation..." << endl;
        
        EvalSettings settings;
        string newModelPath = "";
        string oldModelPath = "";
        
        // Parse command line arguments
        for (int i = 2; i < argc; i++) {
            string arg = argv[i];
            if (arg == "--new" && i + 1 < argc) {
                newModelPath = argv[++i];
            } else if (arg == "--old" && i + 1 < argc) {
                oldModelPath = argv[++i];
            } else if ((arg == "--games" || arg == "-g") && i + 1 < argc) {
                settings.numGames = stoul(argv[++i]);
            } else if ((arg == "--nodes" || arg == "-n") && i + 1 < argc) {
                settings.nodesPerMove = stoul(argv[++i]);
            } else if (arg == "--time" && i + 1 < argc) {
                settings.moveTimeMs = stoi(argv[++i]);
            } else if ((arg == "--temperature" || arg == "--temp" || arg == "-t") && i + 1 < argc) {
                settings.temperature = stof(argv[++i]);
            } else if (arg == "--temp-moves" && i + 1 < argc) {
                settings.temperatureDecayMoves = stoul(argv[++i]);
            } else if (arg == "--verbose" || arg == "-v") {
                settings.verbose = true;
            } else if (arg == "--pgn" && i + 1 < argc) {
                settings.outputPgnPath = argv[++i];
            } else if (arg == "--gui") {
                settings.enableGui = true;
            } else if (arg == "--gui-path" && i + 1 < argc) {
                settings.guiStatePath = argv[++i];
                settings.enableGui = true;
            } else if (arg == "--time-control" && i + 1 < argc) {
                // Deciseconds on all four clocks, as in param-eval. Without it
                // the clock model is off and channels 31 and 63 are a constant
                // team bit -- which makes any comparison of two clock encodings
                // vacuous, since the quantity under test never varies.
                settings.initialTimeDcs = stoi(argv[++i]);
            } else if (arg == "--new-encoding" && i + 1 < argc) {
                if (!parse_time_encoding(argv[++i], settings.player1.timeEncoding)) {
                    return EXIT_FAILURE;
                }
            } else if (arg == "--old-encoding" && i + 1 < argc) {
                if (!parse_time_encoding(argv[++i], settings.player2.timeEncoding)) {
                    return EXIT_FAILURE;
                }
            }
            // Both default to 1.0. Exposed here so a rerun of a pre-2026-08-10
            // gate can reproduce the 0.5/1.5 handicap it was actually measured
            // under, and so a script asking for 1.0 fails loudly on an old
            // binary instead of being silently ignored.
            else if (arg == "--new-attacker-mult" && i + 1 < argc) {
                settings.player1.attackerNodeMultiplier = stof(argv[++i]);
            } else if (arg == "--new-defender-mult" && i + 1 < argc) {
                settings.player1.defenderNodeMultiplier = stof(argv[++i]);
            } else if (arg == "--old-attacker-mult" && i + 1 < argc) {
                settings.player2.attackerNodeMultiplier = stof(argv[++i]);
            } else if (arg == "--old-defender-mult" && i + 1 < argc) {
                settings.player2.defenderNodeMultiplier = stof(argv[++i]);
            }
        }

        // Validate model paths
        if (newModelPath.empty() || oldModelPath.empty()) {
            cerr << "Error: Both --new and --old model paths are required" << endl;
            cerr << "Usage: " << argv[0] << " eval --new <path> --old <path> [options]" << endl;
            return EXIT_FAILURE;
        }

        // An encoding is a claim about how a network was trained, and a wrong
        // claim produces a plausible-looking result rather than an error. State
        // both, every run, in the output that gets pasted into the writeup.
        cout << "  New model encoding: "
             << (settings.player1.timeEncoding == TimeEncoding::Mode::CONTINUOUS
                     ? "continuous" : "binary") << endl;
        cout << "  Old model encoding: "
             << (settings.player2.timeEncoding == TimeEncoding::Mode::CONTINUOUS
                     ? "continuous" : "binary") << endl;
        cout << "  Time control: " << settings.initialTimeDcs << " ds"
             << (settings.initialTimeDcs == 0 ? "  (no clock model -- the margin planes are constant)" : "")
             << endl;
        cout << "  Node multipliers: new att " << settings.player1.attackerNodeMultiplier
             << " / def " << settings.player1.defenderNodeMultiplier
             << ", old att " << settings.player2.attackerNodeMultiplier
             << " / def " << settings.player2.defenderNodeMultiplier << endl;

        run_model_eval(newModelPath, oldModelPath, settings);
        
        return EXIT_SUCCESS;
    }

    // Check for param-eval flag (same model, different parameters)
    if (argc > 1 && string(argv[1]) == "param-eval") {
        cout << "Starting parameter evaluation..." << endl;
        
        EvalSettings settings;
        settings.usePlayerConfigs = true;
        settings.player1.name = "Player1";
        settings.player2.name = "Player2";
        string modelPath = "";
        
        // Parse command line arguments
        for (int i = 2; i < argc; i++) {
            string arg = argv[i];
            if (arg == "--model" && i + 1 < argc) {
                modelPath = argv[++i];
            } else if ((arg == "--games" || arg == "-g") && i + 1 < argc) {
                settings.numGames = stoul(argv[++i]);
            } else if (arg == "--time-control" && i + 1 < argc) {
                // Deciseconds on all four clocks. Omitted or 0 means no clock
                // model: sitting is free and nothing can end on time, which is
                // the baseline arm to measure the clock-aware engine against.
                settings.initialTimeDcs = stoi(argv[++i]);
            } else if (arg == "--encoding" && i + 1 < argc) {
                // One model plays both sides here, so the encoding is a single
                // fact about that .onnx rather than a per-player knob. Without
                // this the default would silently present a continuous-trained
                // net with the binary planes it has never seen -- an A/B that
                // completes and reports a plausible Elo for the wrong reason.
                TimeEncoding::Mode mode;
                if (!parse_time_encoding(argv[++i], mode)) {
                    return EXIT_FAILURE;
                }
                settings.player1.timeEncoding = mode;
                settings.player2.timeEncoding = mode;
            }
            // Player 1 settings
            else if (arg == "--p1-nodes" && i + 1 < argc) {
                settings.player1.nodesPerMove = stoul(argv[++i]);
            } else if (arg == "--p1-time" && i + 1 < argc) {
                settings.player1.moveTimeMs = stoi(argv[++i]);
            } else if (arg == "--p1-batch" && i + 1 < argc) {
                settings.player1.batchSize = stoi(argv[++i]);
            } else if (arg == "--p1-threads" && i + 1 < argc) {
                settings.player1.numSearchThreads = stoi(argv[++i]);
            } else if (arg == "--p1-cpuct" && i + 1 < argc) {
                settings.player1.cpuctInit = stof(argv[++i]);
            } else if (arg == "--p1-fpu" && i + 1 < argc) {
                settings.player1.fpuReduction = stof(argv[++i]);
            } else if (arg == "--p1-name" && i + 1 < argc) {
                settings.player1.name = argv[++i];
            } else if (arg == "--p1-contempt" && i + 1 < argc) {
                settings.player1.drawContempt = stof(argv[++i]);
            } else if (arg == "--p1-pw-coef" && i + 1 < argc) {
                settings.player1.pwCoefficient = stof(argv[++i]);
            } else if (arg == "--p1-pw-exp" && i + 1 < argc) {
                settings.player1.pwExponent = stof(argv[++i]);
            } else if (arg == "--p1-mcgs" && i + 1 < argc) {
                settings.player1.enableMCGS = (stoi(argv[++i]) != 0);
            } else if (arg == "--p1-tt" && i + 1 < argc) {
                settings.player1.enableTranspositions = (stoi(argv[++i]) != 0);
            } else if (arg == "--p1-qweight" && i + 1 < argc) {
                settings.player1.qValueWeight = stof(argv[++i]);
            } else if (arg == "--p1-qveto" && i + 1 < argc) {
                settings.player1.qVetoDelta = stof(argv[++i]);
            } else if (arg == "--p1-clocks" && i + 1 < argc) {
                // 0 makes this player's search clock-blind. Pair with
                // --time-control to get the Phase 4 A/B: same net, same world,
                // one side able to reason about the clock and one not.
                settings.player1.clockAware = (stoi(argv[++i]) != 0);
            } else if (arg == "--p1-attacker-mult" && i + 1 < argc) {
                settings.player1.attackerNodeMultiplier = stof(argv[++i]);
            } else if (arg == "--p1-defender-mult" && i + 1 < argc) {
                settings.player1.defenderNodeMultiplier = stof(argv[++i]);
            } else if (arg == "--p1-alloc" && i + 1 < argc) {
                // fixed = constant --p1-nodes and the flat model cost, i.e. every
                // benchmark before this flag existed. flat/arc spend the clock in
                // nodes and make --p1-nodes dead.
                if (!parse_alloc_mode(argv[++i], settings.player1.allocation)) {
                    return EXIT_FAILURE;
                }
            }
            // Player 2 settings
            else if (arg == "--p2-nodes" && i + 1 < argc) {
                settings.player2.nodesPerMove = stoul(argv[++i]);
            } else if (arg == "--p2-time" && i + 1 < argc) {
                settings.player2.moveTimeMs = stoi(argv[++i]);
            } else if (arg == "--p2-batch" && i + 1 < argc) {
                settings.player2.batchSize = stoi(argv[++i]);
            } else if (arg == "--p2-threads" && i + 1 < argc) {
                settings.player2.numSearchThreads = stoi(argv[++i]);
            } else if (arg == "--p2-cpuct" && i + 1 < argc) {
                settings.player2.cpuctInit = stof(argv[++i]);
            } else if (arg == "--p2-fpu" && i + 1 < argc) {
                settings.player2.fpuReduction = stof(argv[++i]);
            } else if (arg == "--p2-name" && i + 1 < argc) {
                settings.player2.name = argv[++i];
            } else if (arg == "--p2-contempt" && i + 1 < argc) {
                settings.player2.drawContempt = stof(argv[++i]);
            } else if (arg == "--p2-pw-coef" && i + 1 < argc) {
                settings.player2.pwCoefficient = stof(argv[++i]);
            } else if (arg == "--p2-pw-exp" && i + 1 < argc) {
                settings.player2.pwExponent = stof(argv[++i]);
            } else if (arg == "--p2-mcgs" && i + 1 < argc) {
                settings.player2.enableMCGS = (stoi(argv[++i]) != 0);
            } else if (arg == "--p2-tt" && i + 1 < argc) {
                settings.player2.enableTranspositions = (stoi(argv[++i]) != 0);
            } else if (arg == "--p2-qweight" && i + 1 < argc) {
                settings.player2.qValueWeight = stof(argv[++i]);
            } else if (arg == "--p2-qveto" && i + 1 < argc) {
                settings.player2.qVetoDelta = stof(argv[++i]);
            } else if (arg == "--p2-clocks" && i + 1 < argc) {
                settings.player2.clockAware = (stoi(argv[++i]) != 0);
            } else if (arg == "--p2-attacker-mult" && i + 1 < argc) {
                settings.player2.attackerNodeMultiplier = stof(argv[++i]);
            } else if (arg == "--p2-defender-mult" && i + 1 < argc) {
                settings.player2.defenderNodeMultiplier = stof(argv[++i]);
            } else if (arg == "--p2-alloc" && i + 1 < argc) {
                if (!parse_alloc_mode(argv[++i], settings.player2.allocation)) {
                    return EXIT_FAILURE;
                }
            }
            // Common settings
            else if (arg == "--alloc-k" && i + 1 < argc) {
                // Nodes per decisecond. The currency of the game rather than a
                // player's policy, so it is deliberately not a per-player knob.
                settings.nodesPerDecisecond = stoi(argv[++i]);
            } else if (arg == "--verbose" || arg == "-v") {
                settings.verbose = true;
            } else if (arg == "--pgn" && i + 1 < argc) {
                settings.outputPgnPath = argv[++i];
            } else if (arg == "--gui") {
                settings.enableGui = true;
            } else if (arg == "--gui-path" && i + 1 < argc) {
                settings.guiStatePath = argv[++i];
                settings.enableGui = true;
            }
        }
        
        // Validate model path
        if (modelPath.empty()) {
            // Try to find a model in ./networks
            modelPath = findLatestOnnxFile("./networks");
            if (modelPath.empty()) {
                cerr << "Error: --model path is required (or place a model in ./networks)" << endl;
                cerr << "Usage: " << argv[0] << " param-eval --model <path> [options]" << endl;
                return EXIT_FAILURE;
            }
            cout << "Using model: " << modelPath << endl;
        }

        // Same discipline as `eval`: the encoding and the clock arms are the
        // whole experiment, so state them in the output that gets pasted into
        // the writeup rather than trusting the invocation to be remembered.
        cout << "  Model encoding: "
             << (settings.player1.timeEncoding == TimeEncoding::Mode::CONTINUOUS
                     ? "continuous" : "binary") << endl;
        cout << "  Time control: " << settings.initialTimeDcs << " ds"
             << (settings.initialTimeDcs == 0 ? "  (no clock model -- both arms identical)" : "")
             << endl;
        cout << "  Search sees clocks: P1 " << (settings.player1.clockAware ? "yes" : "no")
             << ", P2 " << (settings.player2.clockAware ? "yes" : "no") << endl;
        cout << "  Allocation: P1 " << alloc_mode_name(settings.player1.allocation)
             << ", P2 " << alloc_mode_name(settings.player2.allocation)
             << "  (k = " << settings.nodesPerDecisecond << " nodes/ds)" << endl;
        // 1.0/1.0 means the configured node budget is the budget actually
        // searched. Anything else is a handicap and belongs in the writeup.
        cout << "  Node multipliers: P1 att " << settings.player1.attackerNodeMultiplier
             << " / def " << settings.player1.defenderNodeMultiplier
             << ", P2 att " << settings.player2.attackerNodeMultiplier
             << " / def " << settings.player2.defenderNodeMultiplier << endl;

        // An allocation mode with no clock has no bank to spend and silently
        // degrades to FIXED -- a run that completes and reports a plausible Elo
        // for an arm that was never actually tested. Refuse instead.
        const bool wantsAllocation =
            settings.player1.allocation != TimeAlloc::Mode::FIXED ||
            settings.player2.allocation != TimeAlloc::Mode::FIXED;
        if (wantsAllocation && settings.initialTimeDcs <= 0) {
            cerr << "Error: --p1-alloc/--p2-alloc need a clock; pass --time-control <dcs>"
                 << endl;
            return EXIT_FAILURE;
        }
        if (settings.nodesPerDecisecond <= 0) {
            cerr << "Error: --alloc-k must be positive" << endl;
            return EXIT_FAILURE;
        }

        run_param_eval(modelPath, settings);
        
        return EXIT_SUCCESS;
    }

    UCI uci;
    std::vector<int> deviceIds(deviceCount);
    iota(deviceIds.begin(), deviceIds.end(), 0);

    std::cout << "HiveMind 1.0" << std::endl;

    uci.initializeEngines(deviceIds);
    uci.loop();
}
