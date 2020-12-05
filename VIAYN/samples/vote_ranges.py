from VIAYN.project_types import VoteRange


class BinaryVoteRange(VoteRange):
    def contains(self, value: float) -> bool:
        return value in [0, 1]


class FiveStarVoteRange(VoteRange):
    def contains(self, value: float) -> bool:
        return value in [1, 2, 3, 4, 5]


class ZeroToTenVoteRange(VoteRange):
    def contains(self, value: float) -> bool:
        return 0 <= value <= 10
