from fractions import Fraction

import pytest
from notes import ElectricGuitar, Song, Track
from notes.ears.idmt import idmt_music
from notes.ir import Score


def xml(tmp_path, *, pitch='45', string='2', fret='0', start='0.1234', end='.8234',
        excitation='MU', expression='NO', extra=''):
    path = tmp_path/'annotation.xml'
    path.write_text(f'''<instrumentRecording><transcription><event>
      <onsetSec>{start}</onsetSec><offsetSec>{end}</offsetSec><pitch>{pitch}</pitch>
      <stringNumber>{string}</stringNumber><fretNumber>{fret}</fretNumber>
      <excitationStyle>{excitation}</excitationStyle><expressionStyle>{expression}</expressionStyle>
      </event>{extra}</transcription></instrumentRecording>''')
    return path


def score(path):
    return Song(Track('guitar', ElectricGuitar())(idmt_music(path)), bpm=60).compile()


def test_idmt_preserves_decimal_timing_and_reverses_string_index(tmp_path):
    result = Score.from_dict(score(xml(tmp_path)).to_dict())
    event = result.events[0]
    assert event.time == Fraction(617, 5000) and event.end == Fraction(4117, 5000)
    assert event.pitch == 45 and event.vel == .7
    assert event.param('string') == 5 and event.param('palm_mute') is True


def test_high_string_picked_note(tmp_path):
    event = score(xml(tmp_path, pitch='69', string='6', fret='5', excitation='PK')).events[0]
    assert event.param('string') == 1 and event.param('palm_mute') is False


@pytest.mark.parametrize('kwargs', [dict(pitch='46'), dict(string='0'), dict(fret='25'),
    dict(start='-.1'), dict(end='.1'), dict(excitation='FS'), dict(expression='SL'),
    dict(expression='BE'), dict(expression='DN')])
def test_unsupported_or_inconsistent_annotation_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        idmt_music(xml(tmp_path, **kwargs))


def test_polyphonic_notes_keep_their_independent_offsets(tmp_path):
    extra = '''<event><onsetSec>0.1234</onsetSec><offsetSec>1.0</offsetSec><pitch>64</pitch>
    <stringNumber>6</stringNumber><fretNumber>0</fretNumber>
    <excitationStyle>PK</excitationStyle><expressionStyle>NO</expressionStyle></event>'''
    events = score(xml(tmp_path, extra=extra)).events
    assert len(events) == 2 and events[0].time == events[1].time
    assert {e.end for e in events} == {Fraction(4117, 5000), Fraction(1)}


def test_xml_entity_declaration_rejected(tmp_path):
    path = xml(tmp_path)
    path.write_text('<!DOCTYPE x [<!ENTITY y "z">]>'+path.read_text())
    with pytest.raises(ValueError, match='entities'):
        idmt_music(path)
