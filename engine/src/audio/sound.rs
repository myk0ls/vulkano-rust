use std::path::Path;

use symphonia::core::audio::sample::Sample;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// The file extensions sounds can be loaded from.
pub const SOUND_EXTENSIONS: &[&str] = &["wav", "ogg", "mp3", "flac"];

/// A whole decoded sound, as interleaved stereo samples.
#[derive(Clone, Debug, PartialEq)]
pub struct Sound {
    /// Left, right, left, right, ...
    pub samples: Vec<f32>,
    pub rate: u32,
}

impl Sound {
    /// Make a sound from interleaved samples with any number of channels. Mono is played
    /// in both ears; channels past the second are dropped.
    pub fn from_interleaved(samples: &[f32], channels: usize, rate: u32) -> Self {
        let channels = channels.max(1);
        let samples = samples
            .chunks_exact(channels)
            .flat_map(|frame| if channels == 1 { [frame[0], frame[0]] } else { [frame[0], frame[1]] })
            .collect();
        Self { samples, rate: rate.max(1) }
    }

    pub fn frames(&self) -> usize {
        self.samples.len() / 2
    }

    /// Length in seconds.
    pub fn duration(&self) -> f32 {
        self.frames() as f32 / self.rate as f32
    }

    /// Decode a WAV, OGG Vorbis, MP3 or FLAC file.
    pub fn load(path: &str) -> Result<Self, String> {
        let fail = |what: &str, e: &dyn std::fmt::Display| format!("could not {what} {path}: {e}");
        let file = std::fs::File::open(path).map_err(|e| fail("open", &e))?;
        let stream = MediaSourceStream::new(Box::new(file), Default::default());
        let mut hint = Hint::new();
        if let Some(extension) = Path::new(path).extension().and_then(|e| e.to_str()) {
            hint.with_extension(extension);
        }
        let mut format = symphonia::default::get_probe()
            .probe(&hint, stream, FormatOptions::default(), MetadataOptions::default())
            .map_err(|e| fail("read", &e))?;
        let track = format
            .default_track(TrackType::Audio)
            .ok_or_else(|| format!("{path} has no audio"))?;
        let track_id = track.id;
        let params = track
            .codec_params
            .as_ref()
            .and_then(|p| p.audio())
            .ok_or_else(|| format!("{path} has no audio"))?
            .clone();
        let mut decoder = symphonia::default::get_codecs()
            .make_audio_decoder(&params, &AudioDecoderOptions::default())
            .map_err(|e| fail("decode", &e))?;

        let mut rate = params.sample_rate.unwrap_or(44_100);
        let mut stereo = Vec::new();
        let mut packet_samples: Vec<f32> = Vec::new();
        loop {
            let packet = match format.next_packet() {
                Ok(Some(packet)) => packet,
                Ok(None) => break,
                Err(Error::ResetRequired) => break,
                Err(e) => return Err(fail("read", &e)),
            };
            if packet.track_id != track_id {
                continue;
            }
            match decoder.decode(&packet) {
                Ok(buffer) => {
                    rate = buffer.spec().rate();
                    let channels = buffer.spec().channels().count();
                    packet_samples.resize(buffer.samples_interleaved(), f32::MID);
                    buffer.copy_to_slice_interleaved(&mut packet_samples);
                    stereo.extend(Self::from_interleaved(&packet_samples, channels, rate).samples);
                }
                // A damaged packet is skipped, as players do.
                Err(Error::DecodeError(_)) => {}
                Err(e) => return Err(fail("decode", &e)),
            }
        }
        if stereo.is_empty() {
            return Err(format!("{path} has no audio"));
        }
        Ok(Self { samples: stereo, rate })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 16-bit stereo WAV file of `frames` frames: left rises, right is silent.
    fn wav(frames: u32, rate: u32) -> Vec<u8> {
        let data = frames * 4;
        let mut bytes = Vec::new();
        bytes.extend(b"RIFF");
        bytes.extend((36 + data).to_le_bytes());
        bytes.extend(b"WAVEfmt ");
        bytes.extend(16u32.to_le_bytes());
        bytes.extend(1u16.to_le_bytes()); // PCM
        bytes.extend(2u16.to_le_bytes());
        bytes.extend(rate.to_le_bytes());
        bytes.extend((rate * 4).to_le_bytes());
        bytes.extend(4u16.to_le_bytes());
        bytes.extend(16u16.to_le_bytes());
        bytes.extend(b"data");
        bytes.extend(data.to_le_bytes());
        for i in 0..frames {
            bytes.extend(((i * 100) as i16).to_le_bytes());
            bytes.extend(0i16.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn decodes_a_wav_file() {
        let path = std::env::temp_dir().join(format!("vulkano-sound-{}.wav", std::process::id()));
        std::fs::write(&path, wav(22_050, 22_050)).unwrap();
        let sound = Sound::load(path.to_str().unwrap());
        std::fs::remove_file(&path).unwrap();
        let sound = sound.unwrap();
        assert_eq!(sound.rate, 22_050);
        assert_eq!(sound.frames(), 22_050);
        assert!((sound.duration() - 1.0).abs() < 1e-6);
        assert!((sound.samples[2] - 100.0 / 32768.0).abs() < 1e-6);
        assert_eq!(sound.samples[3], 0.0);
    }

    #[test]
    fn missing_files_are_an_error() {
        assert!(Sound::load("no/such/sound.ogg").is_err());
    }
}
