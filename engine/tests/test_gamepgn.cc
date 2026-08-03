#include <gtest/gtest.h>

#include "../src/rl/gamepgn.h"
#include <sstream>
#include <string>

// PGN headers that assert something about time.
//
// Both were silently wrong once Phase 4 gave games a clock model: TimeControl
// was a hardcoded "180" that nothing ever set, and TimeAdvantage named a single
// team for a whole game even though clocks make it change hands. Neither tag is
// parsed anywhere, which is exactly why they could stay wrong unnoticed -- the
// PGN is evidence, and evidence that lies is worse than none.
class GamePGNTest : public ::testing::Test {
protected:
    static BughouseGamePGN fresh() {
        BughouseGamePGN pgn;
        pgn.whiteTeam = "Alice";
        pgn.blackTeam = "Bob";
        pgn.whiteTeamHadTimeAdvantage = true;
        return pgn;
    }

    static std::string render(const BughouseGamePGN& pgn) {
        std::ostringstream os;
        os << pgn;
        return os.str();
    }

    static bool contains(const std::string& haystack, const std::string& needle) {
        return haystack.find(needle) != std::string::npos;
    }
};

TEST_F(GamePGNTest, NoClockModelKeepsTheTimeAdvantageTag) {
    BughouseGamePGN pgn = fresh();  // initialTimeDcs defaults to 0

    const std::string out = render(pgn);

    // Self-play still has no clock model, so the bit really is fixed for the
    // whole game and the tag is the truth about it.
    EXPECT_TRUE(contains(out, "[TimeAdvantage \"Alice\"]"));
    EXPECT_TRUE(contains(out, "[TimeControl \"180\"]"));
}

TEST_F(GamePGNTest, AClockModelSuppressesTheTimeAdvantageTag) {
    BughouseGamePGN pgn = fresh();
    pgn.initialTimeDcs = 400;

    const std::string out = render(pgn);

    EXPECT_FALSE(contains(out, "TimeAdvantage"))
        << "with clocks running, no single team holds the advantage for the game";
}

TEST_F(GamePGNTest, TimeControlFollowsTheClockModel) {
    BughouseGamePGN pgn = fresh();
    pgn.initialTimeDcs = 400;  // deciseconds

    EXPECT_TRUE(contains(render(pgn), "[TimeControl \"40\"]"))
        << "400 ds is a 40-second game, not the 180 the default claimed";
}

TEST_F(GamePGNTest, TimeControlStillFollowsTheStringWithoutClocks) {
    BughouseGamePGN pgn = fresh();
    pgn.timeControl = "120";

    // The string is only authoritative when there is no clock model to override
    // it, which keeps every pre-Phase 4 caller behaving as before.
    EXPECT_TRUE(contains(render(pgn), "[TimeControl \"120\"]"));
}
