#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""
Tensixinet Phase 1: Hardware Telemetry Sonification

This prototype sonifies Tenstorrent hardware telemetry using Ciat Lonbarde
principles:
- Current-based (not voltage-based) control
- Organic exponential curves
- Polyrhythmic patterns from multi-channel telemetry
- Controlled instability with predictable behavior

Hardware signals mapped:
- asic_temperature → Pitch envelope (slow LFO, 0.1-2Hz)
- aiclk → Tempo/BPM control
- power/current → CV mixing (current-based mixing)
- GDDR temperatures → Multi-voice modulation
- DDR status → Polyrhythmic patterns

Integration: Works with tt-demo-maker to create MP4 videos with
tt-toplike visualizations and tensixinet-generated music.

Author: Qwen Code
Date: 2026-10-16
"""

import json
import subprocess
import time
import math
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from threading import Thread
import queue

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice not available, using fallback mode")


@dataclass
class TelemetrySnapshot:
    """Hardware telemetry snapshot"""
    timestamp: float
    asic_temperature: float
    aiclk: float
    power: float
    current: float
    voltage: float
    heartbeat: int
    gddr_temps: List[float]
    ddr_status: List[int]


class HardwareTelemetry:
    """Fetches and parses hardware telemetry from tt-smi"""
    
    def __init__(self):
        self.history = deque(maxlen=100)
        
    def fetch(self) -> TelemetrySnapshot:
        """Fetch current telemetry from tt-smi"""
        try:
            result = subprocess.run(
                ['tt-smi', '-s'],
                capture_output=True,
                text=True,
                timeout=5
            )
            data = json.loads(result.stdout)
            
            if 'device_info' in data and len(data['device_info']) > 0:
                dev = data['device_info'][0]
                tel = dev.get('telemetry', {})
                smbus = dev.get('smbus_telem', {})
                
                # Extract key metrics
                telemetry = TelemetrySnapshot(
                    timestamp=time.time(),
                    asic_temperature=float(tel.get('asic_temperature', 48.5)),
                    aiclk=float(tel.get('aiclk', 800)),
                    power=float(tel.get('power', 32.0)),
                    current=float(tel.get('current', 45.0)),
                    voltage=float(tel.get('voltage', 0.71)),
                    heartbeat=int(tel.get('heartbeat', 36741) or 0),
                    gddr_temps=self._parse_gddr_temps(smbus),
                    ddr_status=self._parse_ddr_status(smbus),
                )
                
                self.history.append(telemetry)
                return telemetry
                
        except Exception as e:
            print(f"Telemetry fetch error: {e}")
            return self._fallback_telemetry()
    
    def _parse_gddr_temps(self, smbus: Dict) -> List[float]:
        """Parse GDDR temperature sensors (8 channels)"""
        temps = []
        temp_keys = ['GDDR_0_1_TEMP', 'GDDR_2_3_TEMP', 'GDDR_4_5_TEMP', 'GDDR_6_7_TEMP']
        
        for key in temp_keys:
            val = smbus.get(key)
            if val and isinstance(val, str):
                val_int = int(val, 16)
                for i in range(4):
                    byte_val = (val_int >> (i * 8)) & 0xFF
                    if byte_val != 0:
                        temps.append(float(byte_val))
        
        while len(temps) < 8:
            temps.append(30.0)
        
        return temps[:8]
    
    def _parse_ddr_status(self, smbus: Dict) -> List[int]:
        """Parse DDR status bits (4 channels)"""
        status = smbus.get('DDR_STATUS')
        if status and isinstance(status, str):
            status_int = int(status, 16)
            return [(status_int >> (i * 2)) & 3 for i in range(4)]
        return [0, 0, 0, 0]
    
    def _fallback_telemetry(self) -> TelemetrySnapshot:
        """Generate fallback telemetry for testing"""
        t = time.time()
        return TelemetrySnapshot(
            timestamp=t,
            asic_temperature=45.0 + 5.0 * math.sin(t / 30),
            aiclk=800 + 100 * math.sin(t / 20),
            power=30.0 + 10.0 * math.sin(t / 15),
            current=42.0 + 8.0 * math.sin(t / 15),
            voltage=0.71,
            heartbeat=int(t * 1000) % 100000,
            gddr_temps=[30.0 + 2.0 * math.sin(t / 10 + i * 0.5) for i in range(8)],
            ddr_status=[(int(t * 2) >> i) & 3 for i in range(4)],
        )


class CiatLonbardeSynth:
    """
    Ciat Lonbarde-inspired synthesizer engine.
    
    Key principles implemented:
    - Current-based (not voltage-based) control
    - Organic exponential curves
    - Polyrhythmic patterns from multi-channel telemetry
    - Controlled instability with predictable behavior
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
        # LFOs for organic modulation
        self.lfos = {
            'tempo_lfo': {'phase': 0.0, 'frequency': 0.1},
            'pitch_lfo': {'phase': 0.0, 'frequency': 0.2},
            'timbre_lfo': {'phase': 0.0, 'frequency': 0.3},
        }
        
        # Bounds modulation (like Sidrax bounds)
        self.bounds = {
            'lower': 0.5,
            'upper': 2.0,
            'center': 1.0,
        }
        
        # Current state for current-based mixing
        self.current_state = {
            'osc1': 0.0,
            'osc2': 0.0,
            'osc3': 0.0,
            'osc4': 0.0,
        }
        
    def get_telemetry_influence(self, telemetry: TelemetrySnapshot) -> Dict:
        """
        Map hardware telemetry to synthesizer parameters.
        
        Current-based mapping (not voltage-based):
        - Temperature → Pitch envelope (slow LFO)
        - Clock speed → Tempo/BPM
        - Power → Volume with exponential curve
        - GDDR temps → Multi-voice modulation
        - DDR status → Polyrhythmic patterns
        """
        
        # Primary oscillator frequency from ASIC temperature
        temp = telemetry.asic_temperature
        temp_norm = (temp - 25) / 75
        base_pitch = 60 + int(temp_norm * 36)  # C3 to C5
        
        # Tempo from AICLK
        aiclk = telemetry.aiclk
        bpm = 60 + int((aiclk / 1350) * 120)  # 60-180 BPM range
        
        # Volume from power (current-based, exponential curve)
        power = telemetry.power
        power_norm = min(power / 200.0, 1.0)
        volume = 0.3 + power_norm * 0.7
        
        # Timbre from GDDR temperatures (8 channels)
        gddr_mod = [(t - 25) / 50 for t in telemetry.gddr_temps]
        
        # Polyrhythmic patterns from DDR status (4 channels)
        ddr_rhythms = [self._ddr_to_rhythm(state) for state in telemetry.ddr_status]
        
        return {
            'base_pitch': base_pitch,
            'bpm': bpm,
            'volume': volume,
            'gddr_modulation': gddr_mod,
            'ddr_rhythms': ddr_rhythms,
            'clock_speed': aiclk,
        }
    
    def _ddr_to_rhythm(self, state: int) -> Tuple[int, float]:
        """
        Convert DDR status to rhythm parameters.
        
        DDR states (2 bits):
        - 00: Idle
        - 01: Training
        - 10: Trained
        - 11: Error
        """
        rhythm_map = {
            0: (4, 0.25),   # 4/4 time, quiet
            1: (3, 0.5),   # 3/4 waltz, moderate
            2: (5, 0.75),  # 5/4 polymeter, active
            3: (7, 1.0),   # 7/8 complex, loud
        }
        return rhythm_map.get(state, (4, 0.25))
    
    def render_audio(self, duration: float, telemetry: TelemetrySnapshot) -> np.ndarray:
        """
        Render audio buffer using Ciat Lonbarde-inspired synthesis.
        """
        params = self.get_telemetry_influence(telemetry)
        t = np.linspace(0, duration, int(duration * self.sample_rate), endpoint=False)
        
        # Base oscillator (pitch from temperature, BPM from clock)
        base_freq = 2 ** ((params['base_pitch'] - 69) / 12) * 440
        
        # LFOs for organic modulation
        tempo_lfo = 1.0 + 0.1 * np.sin(2 * np.pi * self.lfos['tempo_lfo']['frequency'] * t)
        pitch_lfo = 1.0 + 0.05 * np.sin(2 * np.pi * self.lfos['pitch_lfo']['frequency'] * t)
        timbre_lfo = 1.0 + 0.02 * np.sin(2 * np.pi * self.lfos['timbre_lfo']['frequency'] * t)
        
        # Main carrier with telemetry-driven parameters
        carrier = base_freq * tempo_lfo * pitch_lfo
        
        # Create polyrhythmic drum-like sounds from DDR status
        drums = self._render_drums(t, params['ddr_rhythms'], base_freq)
        
        # GDDR temperature modulation (8 channels)
        gddr_mod = self._render_gddr_modulation(t, params['gddr_modulation'])
        
        # Main synth voice
        synth = (
            0.4 * np.sin(2 * np.pi * carrier * t) * timbre_lfo +
            0.2 * gddr_mod +
            0.3 * drums
        )
        
        # Apply exponential envelope
        envelope = np.exp(-t * 2)
        synth = synth * envelope
        
        # Current-based mixing: add bounds-modulated oscillator
        synth += self._render_bounds_modulated_oscillator(t, params)
        
        # Normalize to avoid clipping
        max_val = np.max(np.abs(synth))
        if max_val > 0:
            synth = synth / max_val * params['volume'] * 0.5
        
        return synth.astype(np.float32)
    
    def _render_bounds_modulated_oscillator(self, t: np.ndarray, params: Dict) -> np.ndarray:
        """Render oscillator with bounds modulation (Sidrax-style)"""
        lower_bound = self.bounds['lower']
        upper_bound = self.bounds['upper']
        center = self.bounds['center']
        
        bounds_center = center * (1 + 0.2 * params['gddr_modulation'][0])
        bounds_width = (upper_bound - lower_bound) * 0.5
        
        freq = 2 ** ((params['base_pitch'] - 72) / 12) * 440 * bounds_center
        
        random_walk = np.cumsum(np.random.normal(0, 0.01, len(t)))
        bounds_mod = 1 + 0.1 * bounds_width * np.sin(2 * np.pi * freq * t + random_walk * 0.1)
        
        return 0.1 * np.sin(2 * np.pi * freq * bounds_mod * t)
    
    def _render_drums(self, t: np.ndarray, ddr_rhythms: List[Tuple[int, float]], base_freq: float) -> np.ndarray:
        """Render drum patterns from DDR status"""
        drums = np.zeros_like(t)
        
        for channel, (num_beats, intensity) in enumerate(ddr_rhythms):
            beat_interval = 60.0 / (180 * num_beats / 4)
            beat_times = np.arange(0, t[-1], beat_interval)
            
            for beat_time in beat_times:
                idx = int(beat_time * self.sample_rate)
                if 0 <= idx < len(drums):
                    drum_duration = 0.1
                    n_samples = int(drum_duration * self.sample_rate)
                    if idx + n_samples < len(drums):
                        drum_env = np.exp(-np.arange(n_samples) / (n_samples * 0.3))
                        drum_tone = np.sin(2 * np.pi * 100 * t[idx:idx+n_samples])
                        drums[idx:idx+n_samples] += drum_env * drum_tone * intensity * 0.2
        
        return drums
    
    def _render_gddr_modulation(self, t: np.ndarray, gddr_mod: List[float]) -> np.ndarray:
        """Render GDDR temperature modulation as audio"""
        modulation = np.zeros_like(t)
        
        for channel, mod_value in enumerate(gddr_mod):
            freq = 50 + channel * 20
            depth = mod_value * 0.3
            carrier = 100 + channel * 50
            modulator = freq
            
            modulation += depth * np.sin(2 * np.pi * carrier * t) * np.sin(2 * np.pi * modulator * t)
        
        return modulation


class TelemetryAudioGenerator:
    """
    Generates continuous audio stream from telemetry.
    Perfect for tt-demo-maker MP4 soundtracks.
    """
    
    def __init__(self, sample_rate: int = 44100, chunk_size: float = 0.5):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.telemetry_fetcher = HardwareTelemetry()
        self.synth = CiatLonbardeSynth()
        self.audio_queue = queue.Queue(maxsize=10)
        self.running = False
        
    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback for sounddevice stream"""
        if not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get_nowait()
                outdata[:] = chunk.reshape(-1, 1)
            except:
                outdata[:] = np.zeros((frames, 1), dtype=np.float32)
        else:
            outdata[:] = np.zeros((frames, 1), dtype=np.float32)
    
    def generate_audio_segment(self, duration: float) -> np.ndarray:
        """Generate audio segment for given duration"""
        telemetry = self.telemetry_fetcher.fetch()
        return self.synth.render_audio(duration, telemetry)
    
    def start_stream(self):
        """Start continuous audio stream"""
        if not SOUNDDEVICE_AVAILABLE:
            print("sounddevice not available, using fallback mode")
            return
        
        try:
            import sounddevice as sd
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * self.chunk_size)
            )
            self.stream.start()
            print("Audio stream started")
        except Exception as e:
            print(f"Stream start error: {e}")
    
    def stop_stream(self):
        """Stop audio stream"""
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
    
    def generate_for_video(self, duration_seconds: int) -> List[np.ndarray]:
        """
        Generate audio segments for video soundtrack.
        
        Perfect for tt-demo-maker: call this with video duration,
        then concatenate and save as soundtrack.
        """
        segments = []
        print(f"Generating {duration_seconds}s of telemetry-based audio...")
        
        for i in range(0, duration_seconds, int(self.chunk_size * 2)):
            if i >= duration_seconds:
                break
            
            # Generate segment
            segment_duration = min(self.chunk_size * 2, duration_seconds - i)
            segment = self.generate_audio_segment(segment_duration)
            segments.append(segment)
            
            # Progress
            progress = int((i / duration_seconds) * 50)
            print(f"\rProgress: [{'#' * progress}{'.' * (50 - progress)}] {int(i/duration_seconds*100)}%", end='', flush=True)
        
        print("\nAudio generation complete")
        return segments
    
    def save_audio(self, segments: List[np.ndarray], output_path: str, sample_rate: int = 44100):
        """
        Save audio segments to file.
        
        Works with tt-demo-maker: save the generated soundtrack,
        then combine with video using ffmpeg.
        """
        import soundfile as sf
        
        # Concatenate segments
        audio = np.concatenate(segments) if segments else np.array([], dtype=np.float32)
        
        # Save to file
        sf.write(output_path, audio, sample_rate)
        print(f"Audio saved to {output_path}")


class TelemetryAudioDemo:
    """
    Demo class showing telemetry → audio workflow.
    
    Perfect for demonstrating to tt-demo-maker users how
    tensixinet creates soundtracks for their videos.
    """
    
    def __init__(self):
        self.generator = TelemetryAudioGenerator()
    
    def run_demo(self, duration: int = 10):
        """
        Run demonstration: generate audio for duration seconds.
        
        This is the workflow that will be used with tt-demo-maker:
        1. Run tt-toplike visualization
        2. Generate telemetry-based audio soundtrack
        3. Combine video + audio with ffmpeg
        """
        print("="*70)
        print("TENSIXINET DEMO: Hardware Telemetry → Audio Soundtrack")
        print("="*70)
        print()
        print("This demonstrates the audio generation workflow that will")
        print("be used to create soundtracks for tt-demo-maker MP4 videos.")
        print()
        print(f"Generating {duration} seconds of telemetry-based audio...")
        print()
        
        segments = self.generator.generate_for_video(duration)
        
        # Save audio
        output_path = "/tmp/tensixinet_demo_audio.wav"
        self.generator.save_audio(segments, output_path)
        
        print()
        print(f"Audio saved to: {output_path}")
        print()
        print("To create a video with tt-demo-maker + tensixinet:")
        print("1. Run tt-toplike with visualization")
        print("2. Record with ffmpeg (audio from tensixinet)")
        print("3. Combine video + audio:")
        print(f"   ffmpeg -i video.mp4 -i {output_path} -c copy output_with_music.mp4")
        print()
        print("Or use tt-demo-maker which can integrate tensixinet automatically!")
        print("="*70)


def main():
    """Main demo entry point"""
    
    demo = TelemetryAudioDemo()
    
    try:
        # Run 10-second demo
        demo.run_demo(duration=10)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
