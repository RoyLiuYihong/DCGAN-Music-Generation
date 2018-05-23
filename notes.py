from pretty_midi import PrettyMIDI, Note, Instrument

import numpy as np
import itertools, collections

DEFAULT_FS = 6
DEFAULT_VELOCITY = 64
DEFAULT_PITCH_RANGE = range(21 + 12, 109 - 12)


def array_to_pm(array,
                fs=DEFAULT_FS,
                velocity=DEFAULT_VELOCITY,
                pitch_range=DEFAULT_PITCH_RANGE):

    pm = PrettyMIDI()
    inst = Instrument(1)
    pm.instruments.append(inst)

    last_notes = {}
    last_state = np.zeros(array.shape[1]) > 0.5

    for i, step in enumerate(array):
        now = i / fs
        step = step > 0.5
        changed = step != last_state
        for pitch in np.where(changed)[0]:
            if step[pitch]:
                last_notes[pitch] = Note(velocity, pitch + pitch_range.start, now, None)
                inst.notes.append(last_notes[pitch])
            else:
                last_notes[pitch].end = now
                del last_notes[pitch]
        last_state = step

    now = (i + 1) / fs
    for note in last_notes.values():
        note.end = now

    return pm


class NoteSeqs:

    def __init__(self, path):
        pm = PrettyMIDI(path)
        pm.remove_invalid_notes()
        self.midi = pm
        self.tempo = pm.estimate_tempo()
        self.beat_times = pm.get_beats()
        self.bar_times = pm.get_downbeats()
        self.end_time = pm.get_end_time()
        self.instruments = pm.instruments


    def get_piano_roll(self,
                       fs=DEFAULT_FS,
                       pitch_range=DEFAULT_PITCH_RANGE):

        # Get note sequence and filter with pitch range
        note_seqs = [
            [note for note in note_seq if note.pitch in pitch_range]
            for note_seq in map(lambda inst: inst.notes, self.instruments)
        ]

        n_keys = len(pitch_range)

        # Merge notes with trimming
        notes = list(itertools.chain(*note_seqs))
        notes.sort(key=lambda note: note.start)
        last_notes = {}
        for note in notes:
            if note.pitch in last_notes:
                note.end = max(note.end, last_notes[note.pitch].end)
                last_note = last_notes[note.pitch]
                if last_note.start == note.start:
                    # Combine two notes if it they have same start times
                    note.velocity = max(note.velocity, last_note.velocity)
                    del last_notes[note.pitch], last_note
                elif last_notes[note.pitch].end > note.start:
                    # Last note ends after current note starts, trim it
                    last_notes[note.pitch].end = note.start
            last_notes[note.pitch] = note

        start_time, end_time = notes[0].start, notes[-1].end
        step_length = 1 / fs
        step_times = np.arange(start_time, end_time, step_length)
        n_steps = step_times.size

        note_pitchs = [note.pitch - pitch_range.start for note in notes]

        note_on_steps = np.searchsorted(step_times,
                                        [note.start for note in notes],
                                        side='right') - 1
        
        note_off_steps = np.searchsorted(step_times,
                                         [note.end for note in notes],
                                         side='right') - 1
        
        array = np.zeros([n_steps, n_keys])
        for note in zip(note_on_steps, note_off_steps, note_pitchs):
            start_step, end_step, pitch = note
            # Draw line on the array
            array[start_step:end_step, pitch] = 1.
            if end_step - start_step > 1:
                array[end_step - 1, pitch] = 0.
        
        return array

        
if __name__ == '__main__':
    import os, random
    from PIL import Image

    ns = NoteSeqs('bach_846.mid')
    piano_roll = ns.get_piano_roll()
    img = Image.fromarray((piano_roll.T * 255).astype(np.uint8))
    img.save(open('piano_roll_image.png', 'wb'))
    array_to_pm(piano_roll).write('out.mid')
