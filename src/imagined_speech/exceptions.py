"""Explicit exception hierarchy for the Imagined Speech Recognition package (Rule 6)."""


class ImaginedSpeechError(Exception):
    """Base exception for all errors raised by the imagined_speech package."""


class ConfigurationError(ImaginedSpeechError):
    """Raised when configuration values, paths, or parameters are invalid or missing."""


class DatasetDiscoveryError(ImaginedSpeechError):
    """Raised when dataset root or expected task directories cannot be discovered."""


class LabelParseError(ImaginedSpeechError):
    """Raised when a recording filename cannot be parsed into a valid task label."""


class EDFValidationError(ImaginedSpeechError):
    """Raised when an EDF file is corrupted, unreadable, or fails structural validation."""


class SignalQualityError(ImaginedSpeechError):
    """Raised when signal-quality input validation or analysis encounters a fatal error."""

