"""ELO rating system utilities."""
from __future__ import annotations
from dataclasses import dataclass, field


def expected_score(rating_a: float, rating_b: float) -> float:
    """Return expected score for player A against player B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update_rating(rating: float, expected: float, score: float, k: float = 32) -> float:
    """Update a single rating given expected score and actual score."""
    return rating + k * (score - expected)


@dataclass
class EloRatingSystem:
    """Simple in-memory ELO rating system."""
    k: float = 32
    ratings: dict[str, float] = field(default_factory=dict)

    def add_player(self, name: str, rating: float = 1000) -> None:
        self.ratings[name] = rating

    def record_match(self, player_a: str, player_b: str, score_a: float) -> tuple[float, float]:
        """Record a match and update ratings.

        Parameters
        ----------
        player_a, player_b: str
            Player identifiers. Players must have been added previously.
        score_a: float
            Score for player A (1.0 win, 0.5 draw, 0.0 loss).
        """
        ra = self.ratings[player_a]
        rb = self.ratings[player_b]
        ea = expected_score(ra, rb)
        eb = expected_score(rb, ra)
        self.ratings[player_a] = update_rating(ra, ea, score_a, self.k)
        self.ratings[player_b] = update_rating(rb, eb, 1 - score_a, self.k)
        return self.ratings[player_a], self.ratings[player_b]
