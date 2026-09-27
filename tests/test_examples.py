"""Every example script must keep compiling to a non-empty score as the library evolves."""

from pathlib import Path

import pytest

from notes.cli import load_song

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("*.py"))


@pytest.mark.parametrize("script", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_example_compiles(script):
    score = load_song(script).compile()
    assert score.events and score.tracks and score.length > 0


def test_first_groove_follows_the_brief():
    score = load_song(Path(__file__).parent.parent / "examples" / "first_groove.py").compile()
    bar = score.beats_per_bar
    assert [m.name for m in score.markers] == ["intro", "verse", "chorus", "chorus", "outro"]
    intro_drums = [e for e in score.events_of("drums") if e.time < 4 * bar]
    assert intro_drums and not any(3 * bar <= e.time < 4 * bar for e in intro_drums)  # bar 4 muted
    assert score.track("lead").instrument["type"] == "electric_guitar"
