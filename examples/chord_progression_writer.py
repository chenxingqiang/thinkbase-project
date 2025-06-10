from typing import List, Optional
from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole
from agentflow.ell.stores import SQLStore
from midiutil.MidiFile import MIDIFile
import pygame
import time

# Get singleton instance
ell = ELLIntegration()

CHORD_FORMAT = "| Chord | Chord | ... |"

@ell.with_ell(mode="simple")
async def write_a_chord_progression_for_song(genre: Optional[str], key: Optional[str]) -> str:
    """Write a chord progression for a song."""
    # Create system message
    system_message = Message(
        role=MessageRole.SYSTEM,
        content="""You are a world class music theorist and composer. Your goal is to write chord progressions to songs given parameters.

Instructions:
1. Write chord progressions that are fully featured and compositionally sound
2. Use advanced chords when appropriate (including 13 chords and complex chords)
3. Only output the chord progression in the format: | C | Am | F | G |
4. Do not provide any additional text or explanations
5. Each chord should be separated by | symbols
6. Use standard chord notation (e.g., C, Am, F7, Cmaj7, etc.)""",
        metadata={
            "type": "text",
            "format": "plain"
        }
    )
    
    # Create user message
    user_message = Message(
        role=MessageRole.USER,
        content=f"Write a chord progression for a song {'in ' + genre if genre else ''} {'in the key of ' + key if key else ''}. Only output the chord progression in the format: | C | Am | F | G |",
        metadata={
            "type": "text",
            "format": "plain"
        }
    )
    
    # Process messages
    await ell.process_message(system_message)
    response = await ell.process_message(user_message)
    
    # Return the response
    if isinstance(response, Message):
        return str(response.content)
    elif isinstance(response, dict):
        return str(response.get("content", ""))
    else:
        return str(response)

@ell.with_ell(mode="simple")
async def parse_chords_to_midi(chords: List[str]) -> str:
    """Convert chord symbols to MIDI notes."""
    # Create system message
    system_message = Message(
        role=MessageRole.SYSTEM,
        content="""You are MusicGPT, an expert at converting chord symbols to MIDI note numbers.

Instructions:
1. Convert each chord to its component MIDI note numbers
2. Each chord should be represented as comma-separated MIDI note numbers
3. Each chord should be on a new line
4. Only output the MIDI note numbers, no additional text
5. Use standard MIDI note numbers (e.g., middle C is 60)
6. Include all notes in the chord (root, third, fifth, and any extensions)""",
        metadata={
            "type": "text",
            "format": "plain"
        }
    )
    
    # Create user message
    user_message = Message(
        role=MessageRole.USER,
        content="""Convert these chords to MIDI note numbers. Only output the numbers in the format:
60,64,67
62,65,69
etc.

Chords:
{}""".format('\n'.join(chord.strip() for chord in chords)),
        metadata={
            "type": "text",
            "format": "plain"
        }
    )
    
    # Process messages
    await ell.process_message(system_message)
    response = await ell.process_message(user_message)
    
    # Return the response
    if isinstance(response, Message):
        return str(response.content)
    elif isinstance(response, dict):
        return str(response.get("content", ""))
    else:
        return str(response)

def create_midi_file(parsed_chords, output_file="chord_progression.mid"):
    """Create a MIDI file from parsed chords."""
    midi = MIDIFile(1)
    track = 0
    time = 0
    midi.addTrackName(track, time, "Chord Progression")
    midi.addTempo(track, time, 60)  # Slower tempo (60 BPM)

    # Set the instrument to Rhodes Piano (MIDI program 4)
    midi.addProgramChange(track, 0, time, 4)

    for chord in parsed_chords:
        notes = [int(note) for note in chord.split(',')]
        for note in notes:
            midi.addNote(track, 0, note, time, 2, 80)  # Longer duration (2 beats) and slightly lower velocity
        time += 2  # Move to the next chord after 2 beats

    with open(output_file, "wb") as output_file:
        midi.writeFile(output_file)

def play_midi_file(file_path):
    """Play a MIDI file using pygame."""
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(1)

if __name__ == "__main__":
    # Initialize ELL
    ell.configure({
        "enabled": True,
        "tracking_enabled": True,
        "store": "./logdir",
        "verbose": True,
        "autocommit": True,
        "model": "gpt-4",
        "default_model": "gpt-4",
        "temperature": 0.5,
        "max_tokens": 1000
    })
    
    # Get user input
    genre = input("Enter the genre of the song (or press Enter to skip): ").strip() or None
    key = input("Enter the key of the song (or press Enter to skip): ").strip() or None

    # Run the example
    import asyncio
    
    # Generate chord progression
    progression = asyncio.run(write_a_chord_progression_for_song(genre=genre, key=key))
    print("\nGenerated Chord Progression:")
    print(progression)
    
    # Parse chords to MIDI
    parsed_chords = asyncio.run(parse_chords_to_midi([chord for chord in progression.split("|") if chord.strip()]))
    print("\nParsed MIDI Notes:")
    print(parsed_chords)
    
    # Create and play MIDI file
    midi_file = "chord_progression.mid"
    create_midi_file(parsed_chords.split('\n'), midi_file)
    print(f"\nMIDI file created: {midi_file}")
    
    print("\nPlaying chord progression...")
    play_midi_file(midi_file)
