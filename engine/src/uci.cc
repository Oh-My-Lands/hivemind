#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <sstream>
#include <vector>

#include <memory>
#include "onnx_utils.h"
#include "planes.h"
#include "utils.h"

#include "uci.h"

using namespace std;

UCI::UCI() : mainSearchThread(nullptr) {

}

UCI::~UCI() {
    // Must signal before joining: an infinite search never ends on its own.
    if (agent) agent->set_is_running(false);
    join_search();
    // Engines will automatically be cleaned up when the vector is destroyed.
}

void UCI::join_search() {
    if (mainSearchThread) {
        if (mainSearchThread->joinable()) {
            mainSearchThread->join();
        }
        delete mainSearchThread;
        mainSearchThread = nullptr;
    }
    ongoingSearch = false;
}

void UCI::initializeEngines(const std::vector<int>& deviceIds) {
    // Clear any existing engines.
    engines.clear();

    // Automatically find the latest ONNX file in the networks directory.
    const std::string onnxFile = findLatestOnnxFile("./networks");
    if (onnxFile.empty()) {
        std::cerr << "Error: No ONNX file found in ./networks directory." << std::endl;
        return;
    }
    // For each device ID, create a new Engine, load the network, and store it.
    for (int deviceId : deviceIds) {
        const std::string engineFile = getEnginePath(onnxFile, "fp16", SearchParams::BATCH_SIZE, deviceId, "v1");
        
        // Create a new engine instance on the given GPU.
        auto enginePtr = std::make_unique<Engine>(deviceId);
        
        // Attempt to load the network (build or deserialize).
        if (!enginePtr->loadNetwork(onnxFile, engineFile)) {
            std::cerr << "Error: Failed to load engine on device " << deviceId << std::endl;
        } else {
            engines.push_back(std::move(enginePtr));
        }
    }

    // Create the single-threaded Agent
    agent = std::make_unique<Agent>();
}


void UCI::stop() {
    if (!ongoingSearch) return;

    // Signal only. Joining here would block the reader loop until the search
    // unwinds and printed its bestmove, which for an analysis client means
    // `isready` and the next `go` stall behind it. The thread is reaped lazily
    // by the next go() or by the destructor.
    agent->set_is_running(false);
    ongoingSearch = false;
}

void UCI::ucinewgame() {
    // A new game invalidates the retained subtrees; without this the next
    // search can reuse a tree built for an unrelated position.
    if (agent) {
        agent->set_is_running(false);
        join_search();
        agent->clear_tree_reuse();
    }
}

void UCI::position(istringstream& is) {
    std::string token;
    is >> token;
    
    // Set the board position
    if (token == "startpos") {
        // Use a predefined starting FEN for the initial position.
        board.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1|rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }
    else if (token == "fen") {
        // Build the FEN string until we hit "moves" or end of stream
        std::string fen;
        while (is >> token && token != "moves") {
            fen += token + " ";
        }
        board.set(fen);
        
        if (token == "moves") {
            is.seekg(-6, std::ios_base::cur);  
        }
    }
    else {
        return;
    }
    
    if (is >> token && token == "moves") {
        // Parse move list (if any)
        int moveCount = 0;
        while (is >> token) {
            if (token.empty() || token[0] < '1' || token[0] > '2') {
                std::cerr << "Error: Invalid board indicator in move '" << token 
                          << "' at move " << (moveCount + 1) << std::endl;
                break;
            }
            int boardNum = token[0] - '1'; // '1' becomes 0, '2' becomes 1.
            std::string moveStr = token.substr(1); // Extract move string without board indicator
            Stockfish::Move m = Stockfish::UCI::to_move(*board.pos[boardNum], moveStr);
            if (m == Stockfish::MOVE_NONE) {
                std::cerr << "Error: Invalid move '" << moveStr << "' on board " 
                          << (boardNum + 1) << " at move " << (moveCount + 1) << std::endl;
                std::cerr << "       Current FEN: " << board.fen(boardNum) << std::endl;
                std::cerr << "       Legal moves: ";
                auto legalMoves = board.legal_moves(boardNum);
                for (const auto& lm : legalMoves) {
                    std::cerr << board.uci_move(boardNum, lm) << " ";
                }
                std::cerr << std::endl;
                break;  // Stop if an invalid move is encountered.
            }
            board.push_move(boardNum, m);
            moveCount++;
        }
    }
}

void UCI::go(std::istringstream& is) {
    std::string token;
    int moveTime = 0;
    size_t nodes = 0;
    bool infinite = false;

    // Parse go parameters
    while (is >> token) {
        if (token == "movetime") {
            is >> moveTime;
        } else if (token == "nodes") {
            is >> nodes;
        } else if (token == "infinite") {
            infinite = true;
        }
    }

    // Stop and reap any previous search before starting a new one. The signal
    // has to come first -- an infinite search would otherwise never return and
    // the join would deadlock.
    agent->set_is_running(false);
    join_search();

    ongoingSearch = true;
    agent->set_is_running(true);

    // Ensure that engines have been initialized.
    if (engines.empty()) {
        std::cerr << "Error: No engines have been initialized!" << std::endl;
        return;
    }

    // Build a vector of raw Engine pointers from the unique_ptr collection.
    std::vector<Engine*> enginePtrs;
    enginePtrs.reserve(engines.size());
    for (const auto& eng : engines) {
        enginePtrs.push_back(eng.get());
    }

    // Build search options based on what was specified
    SearchOptions opts;
    if (infinite) {
        opts = SearchOptions::uci_infinite(multiPV, analysisBoard);
    } else if (nodes > 0) {
        opts = SearchOptions::uci(static_cast<int>(nodes), multiPV, analysisBoard);
        opts.moveTimeMs = 0;  // Node-based search
        opts.targetNodes = nodes;
    } else if (moveTime > 0) {
        opts = SearchOptions::uci(moveTime, multiPV, analysisBoard);
    } else {
        // Default to 1 second if nothing specified
        opts = SearchOptions::uci(1000, multiPV, analysisBoard);
    }
    
    // Launch the search thread
    mainSearchThread = new std::thread([this, enginePtrs, opts]() {
        agent->run_search(board, enginePtrs, teamSide, teamHasTimeAdvantage, opts);
    });
}

void UCI::setoption(std::istringstream& is) {
    std::string token;
    is >> token; 
    if (token != "name") return;
    std::string name;
    is >> name;
    is >> token; 
    if (token != "value") return;
    std::string value;
    is >> value;
    if (name == "Hash") {
        // Parse hash size in MB (1 - 33554432 MB)
        size_t sizeMB = std::stoull(value);
        
        // Set hash size via Agent (which owns the transposition table)
        if (agent) {
            agent->setHashSize(sizeMB);
            std::cout << "info string Hash table set to " << sizeMB << " MB" << std::endl;
        }
    } else if (name == "MultiPV") {
        int mpv = std::stoi(value);
        if (mpv >= 1 && mpv <= 500) {
            multiPV = mpv;
            std::cout << "info string MultiPV set to " << multiPV << std::endl;
        }
    } else if (name == "AnalysisBoard") {
        // Which board's moves MultiPV lines are grouped by. Distinct from Team,
        // which is a colour -- the engine plays both boards, but a human
        // analysing is sitting at one of them.
        int b = std::stoi(value);
        if (b == 1 || b == 2) {
            analysisBoard = (b == 1) ? BOARD_A : BOARD_B;
            std::cout << "info string AnalysisBoard set to " << b << std::endl;
        }
    } else if (name == "Team") {
        if (value == "white") {
            teamSide = Stockfish::WHITE;
        } else if (value == "black") {
            teamSide = Stockfish::BLACK;
        }
    } else if (name == "Mode") {
        if (value == "sit") {
            teamHasTimeAdvantage = true;
        } else if (value == "go") {
            teamHasTimeAdvantage = false;
        }
    } else if (name == "TimeEncoding") {
        // Must match the network loaded. A network trained on one encoding
        // reads the other as garbage -- BINARY's 0/1 lands mid-range for a
        // CONTINUOUS net, and CONTINUOUS's negatives are off the end of
        // BINARY's. There is no way to detect the mismatch from the weights,
        // so this is stated rather than inferred.
        if (value == "binary") {
            timeEncoding = TimeEncoding::Mode::BINARY;
        } else if (value == "continuous") {
            timeEncoding = TimeEncoding::Mode::CONTINUOUS;
        } else {
            std::cout << "info string TimeEncoding ignored: expected binary or continuous"
                      << std::endl;
            return;
        }
        std::cout << "info string TimeEncoding set to " << value << std::endl;
    } else if (name == "Clocks") {
        // "Clocks" takes four deciseconds: A-White A-Black B-White B-Black.
        // `value` above only captured the first token, so re-read the rest.
        int parsed[4] = {0, 0, 0, 0};
        parsed[0] = std::atoi(value.c_str());
        bool ok = true;
        for (int i = 1; i < 4; ++i) {
            std::string tok;
            if (!(is >> tok)) { ok = false; break; }
            parsed[i] = std::atoi(tok.c_str());
        }
        if (!ok) {
            std::cout << "info string Clocks ignored: expected four integers "
                         "(A-White A-Black B-White B-Black, deciseconds)" << std::endl;
            return;
        }
        board.set_clocks(parsed[0], parsed[1], parsed[2], parsed[3]);
        std::cout << "info string Clocks set to " << parsed[0] << " " << parsed[1]
                  << " " << parsed[2] << " " << parsed[3] << " (ds)" << std::endl;
    }
}

void UCI::send_uci_response() {
    cout << "id name hivemind" << endl;
    cout << "id author aminwoo\n" << endl;
    cout << "option name Hash type spin default 16 min 1 max 33554432" << endl;
    cout << "option name MultiPV type spin default 1 min 1 max 500" << endl;
    cout << "option name AnalysisBoard type spin default 1 min 1 max 2" << endl;
    cout << "option name Team type combo default white var white var black" << endl;
    cout << "option name Mode type combo default go var sit var go" << endl;
    cout << "option name TimeEncoding type combo default binary var binary var continuous" << endl;
    // Four deciseconds: A-White A-Black B-White B-Black. Unset means no clock
    // model, and the engine falls back to Mode's single global bit.
    cout << "option name Clocks type string default" << endl;
    cout << "uciok" << endl;
}

void UCI::policy() {
    if (engines.empty()) {
        cerr << "Error: No engines have been initialized!" << endl;
        return;
    }

    // Allocate inference buffers
    float* obs = new float[SearchParams::BATCH_SIZE * NB_INPUT_VALUES()];
    float* value = new float[SearchParams::BATCH_SIZE];
    float* piA = new float[SearchParams::BATCH_SIZE * NB_POLICY_VALUES()];
    float* piB = new float[SearchParams::BATCH_SIZE * NB_POLICY_VALUES()];

    // Convert board to planes
    board_to_planes(board, obs, teamSide,
                    plane_margins(board, teamSide, teamHasTimeAdvantage, timeEncoding));

    // Run inference
    Engine* engine = engines[0].get();
    if (!engine->runInference(obs, value, piA, piB)) {
        cerr << "Inference failed" << endl;
        delete[] obs;
        delete[] value;
        delete[] piA;
        delete[] piB;
        return;
    }

    cout << "Value: " << value[0] << endl;
    cout << endl;

    // Board A policy
    cout << "Board A (" << board.fen(BOARD_A) << "):" << endl;
    if (board.side_to_move(BOARD_A) == teamSide) {
        vector<Stockfish::Move> actionsA = board.legal_moves(BOARD_A);
        actionsA.push_back(Stockfish::MOVE_NONE);  // Add sit option
        vector<float> priorsA = get_normalized_probability(piA, actionsA, BOARD_A, board);
        
        // Sort by probability (descending)
        vector<size_t> indices = argsort(priorsA);
        for (size_t idx : indices) {
            string moveStr = (actionsA[idx] == Stockfish::MOVE_NONE) 
                            ? "pass" : board.uci_move(BOARD_A, actionsA[idx]);
            cout << "  " << moveStr << ": " << priorsA[idx] << endl;
        }
    } else {
        cout << "  (not our turn)" << endl;
    }
    cout << endl;

    // Board B policy
    cout << "Board B (" << board.fen(BOARD_B) << "):" << endl;
    if (board.side_to_move(BOARD_B) == ~teamSide) {
        vector<Stockfish::Move> actionsB = board.legal_moves(BOARD_B);
        actionsB.push_back(Stockfish::MOVE_NONE);  // Add sit option
        vector<float> priorsB = get_normalized_probability(piB, actionsB, BOARD_B, board);
        
        // Sort by probability (descending)
        vector<size_t> indices = argsort(priorsB);
        for (size_t idx : indices) {
            string moveStr = (actionsB[idx] == Stockfish::MOVE_NONE) 
                            ? "pass" : board.uci_move(BOARD_B, actionsB[idx]);
            cout << "  " << moveStr << ": " << priorsB[idx] << endl;
        }
    } else {
        cout << "  (not our turn)" << endl;
    }

    delete[] obs;
    delete[] value;
    delete[] piA;
    delete[] piB;
}


void UCI::loop() {
    string token, cmd;

    do {
        if (!getline(cin, cmd)) // Block here waiting for input or EOF
            cmd = "quit";

        istringstream is(cmd);

        token.clear(); // Avoid a stale if getline() returns empty or blank line
        is >> skipws >> token;

        if (token == "uci")             send_uci_response();
        else if (token == "isready")    cout << "readyok" << endl;
        else if (token == "go")         go(is);
        else if (token == "setoption")  setoption(is);
        else if (token == "position")   position(is);
        else if (token == "stop")       stop();
        else if (token == "ucinewgame") ucinewgame();
        else if (token == "policy")     policy();

    } while (token != "quit"); // Command line args are one-shot
}