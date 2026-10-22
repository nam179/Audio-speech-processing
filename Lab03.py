"""Vietnamese isolated-command recognition with dynamic MFCCs and Gaussian HMMs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import pickle
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct
from scipy.io import wavfile
from scipy.special import logsumexp
from scipy.signal import resample_poly
from scipy.spatial.distance import cdist


SAMPLE_RATE = 16_000
N_MFCC = 13
FEATURE_DIM = 39
LABELS = ("mở", "đóng", "lên", "xuống", "dừng")
ALIASES = {
    "mở": {"mở", "mo", "open"},
    "đóng": {"đóng", "dong", "close"},
    "lên": {"lên", "len", "up"},
    "xuống": {"xuống", "xuong", "down"},
    "dừng": {"dừng", "dung", "stop"},
}


def canonical_label(name: str) -> str | None:
    key = name.casefold().strip()
    for label, aliases in ALIASES.items():
        if key in aliases:
            return label
    return None


def read_audio(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a PCM WAV, convert it to mono float audio, and resample if needed."""
    sr, samples = wavfile.read(path)
    original_dtype = samples.dtype
    if samples.ndim == 2:
        samples = samples.astype(np.float64).mean(axis=1)
    else:
        samples = samples.astype(np.float64)
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        scale = float(max(abs(info.min), info.max))
        samples /= scale
    if sr != target_sr:
        divisor = math.gcd(int(sr), target_sr)
        samples = resample_poly(samples, target_sr // divisor, int(sr) // divisor)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError(f"Audio is empty or contains invalid values: {path}")
    return samples.astype(np.float64, copy=False)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 26) -> np.ndarray:
    def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(20.0), hz_to_mel(sr / 2.0), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mel_points) / sr).astype(int)
    bank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1 : m + 2]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(left, min(center, bank.shape[1])):
            bank[m - 1, k] = (k - left) / (center - left)
        for k in range(center, min(right, bank.shape[1])):
            bank[m - 1, k] = (right - k) / (right - center)
    return bank


def _delta(features: np.ndarray, width: int = 2) -> np.ndarray:
    """Estimate dynamic coefficients with the standard symmetric regression."""
    padded = np.pad(features, ((width, width), (0, 0)), mode="edge")
    denominator = 2.0 * sum(i * i for i in range(1, width + 1))
    result = np.zeros_like(features)
    for i in range(1, width + 1):
        result += i * (padded[width + i : width + i + len(features)]
                       - padded[width - i : width - i + len(features)])
    return result / denominator


def extract_features_from_audio(
    audio: np.ndarray, sr: int = SAMPLE_RATE, dynamic: bool = True
) -> np.ndarray:
    """Return utterance-CMVN MFCC (13-D) or MFCC+delta+delta-delta (39-D)."""
    audio = np.asarray(audio, dtype=np.float64)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak == 0:
        raise ValueError("Cannot extract features from silent audio")

    # Trim leading/trailing silence at 30 dB below the utterance peak.
    indices = np.flatnonzero(np.abs(audio) >= peak * 10 ** (-30.0 / 20.0))
    if indices.size == 0:
        raise ValueError("No speech detected in audio")
    audio = audio[indices[0] : indices[-1] + 1]
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    frame_length = int(round(0.025 * sr))
    hop_length = int(round(0.010 * sr))
    n_fft = 512
    if audio.size < frame_length:
        audio = np.pad(audio, (0, frame_length - audio.size))
    starts = np.arange(0, max(1, audio.size - frame_length + 1), hop_length)
    frames = np.stack([audio[start : start + frame_length] for start in starts])
    window = np.hamming(frame_length)
    spectrum = np.fft.rfft(frames * window, n=n_fft, axis=1)
    power = (np.abs(spectrum) ** 2) / n_fft
    mel_energy = power @ _mel_filterbank(sr, n_fft).T
    static = dct(np.log(np.maximum(mel_energy, 1e-10)), type=2, axis=1, norm="ortho")[
        :, :N_MFCC
    ]
    features = (
        np.concatenate((static, _delta(static), _delta(_delta(static))), axis=1)
        if dynamic
        else static
    )
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return ((features - mean) / np.maximum(std, 1e-8)).astype(np.float64)


def extract_features(path: Path, dynamic: bool = True) -> np.ndarray:
    return extract_features_from_audio(read_audio(path), dynamic=dynamic)


def collect_split(directory: Path) -> dict[str, list[Path]]:
    utterances = {label: [] for label in LABELS}
    if not directory.exists():
        return utterances
    for child in sorted(directory.iterdir()):
        if child.is_dir():
            label = canonical_label(child.name)
            if label is None:
                warnings.warn(f"Ignoring unrecognized label directory: {child}")
                continue
            utterances[label].extend(sorted(child.glob("*.wav")))
        elif child.is_file() and child.suffix.casefold() == ".wav":
            label = canonical_label(child.stem)
            if label is None:
                warnings.warn(f"Ignoring WAV with unrecognized label: {child}")
                continue
            utterances[label].append(child)
    return utterances


def collect_flat_wavs(directory: Path) -> dict[str, list[Path]]:
    utterances = {label: [] for label in LABELS}
    for path in sorted(directory.glob("*.wav")):
        label = canonical_label(path.stem)
        if label is None:
            warnings.warn(f"Ignoring WAV with unrecognized label: {path}")
            continue
        utterances[label].append(path)
    return utterances


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_no_split_leakage(
    train: dict[str, list[Path]], test: dict[str, list[Path]]
) -> None:
    train_hashes = {_sha256(path): path for files in train.values() for path in files}
    duplicated = [
        (train_hashes[_sha256(path)], path)
        for files in test.values()
        for path in files
        if _sha256(path) in train_hashes
    ]
    if duplicated:
        pairs = ", ".join(f"{a} == {b}" for a, b in duplicated)
        raise ValueError(f"Train/test leakage: identical audio appears in both splits: {pairs}")


class ConvergenceMonitor:
    def __init__(self) -> None:
        self.history: list[float] = []
        self.iter = 0
        self.converged = False


class GaussianHMM:
    """Diagonal/full Gaussian HMM trained with constrained Baum-Welch."""

    def __init__(
        self,
        n_components: int,
        covariance_type: str = "diag",
        n_iter: int = 80,
        tol: float = 1e-3,
        min_covar: float = 1e-2,
    ) -> None:
        if covariance_type not in ("diag", "full"):
            raise ValueError("covariance_type must be 'diag' or 'full'")
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.min_covar = min_covar
        self.monitor_ = ConvergenceMonitor()
        self.startprob_ = np.zeros(n_components)
        self.startprob_[0] = 1.0
        self.transmat_ = np.zeros((n_components, n_components))
        self.allowed_transitions_ = np.zeros_like(self.transmat_, dtype=bool)
        for state in range(n_components - 1):
            self.allowed_transitions_[state, state : state + 2] = True
            self.transmat_[state, state] = 0.7
            self.transmat_[state, state + 1] = 0.3
        self.allowed_transitions_[-1, -1] = True
        self.transmat_[-1, -1] = 1.0
        self.means_: np.ndarray | None = None
        self.covars_: np.ndarray | None = None

    def _log_emissions(self, sequence: np.ndarray) -> np.ndarray:
        if self.means_ is None or self.covars_ is None:
            raise RuntimeError("HMM must be fitted before scoring")
        frame_count, dimensions = sequence.shape
        result = np.empty((frame_count, self.n_components), dtype=np.float64)
        for state in range(self.n_components):
            delta = sequence - self.means_[state]
            if self.covariance_type == "diag":
                variance = self.covars_[state]
                result[:, state] = -0.5 * (
                    dimensions * math.log(2.0 * math.pi)
                    + np.log(variance).sum()
                    + ((delta * delta) / variance).sum(axis=1)
                )
            else:
                covariance = self.covars_[state]
                chol = np.linalg.cholesky(covariance)
                whitened = np.linalg.solve(chol, delta.T)
                log_determinant = 2.0 * np.log(np.diag(chol)).sum()
                result[:, state] = -0.5 * (
                    dimensions * math.log(2.0 * math.pi)
                    + log_determinant
                    + (whitened * whitened).sum(axis=0)
                )
        return result

    def _forward_backward(
        self, sequence: np.ndarray, need_expectations: bool = False
    ) -> tuple[float, np.ndarray | None, np.ndarray | None]:
        emissions = self._log_emissions(sequence)
        log_transition = np.full_like(self.transmat_, -np.inf)
        positive = self.transmat_ > 0
        log_transition[positive] = np.log(self.transmat_[positive])
        log_start = np.full(self.n_components, -np.inf)
        positive_start = self.startprob_ > 0
        log_start[positive_start] = np.log(self.startprob_[positive_start])
        log_self_transition = np.diag(log_transition)
        log_next_transition = np.diag(log_transition, k=1)

        alpha = np.empty_like(emissions)
        alpha[0] = log_start + emissions[0]
        for t in range(1, len(sequence)):
            advance = np.full(self.n_components, -np.inf)
            advance[1:] = alpha[t - 1, :-1] + log_next_transition
            alpha[t] = emissions[t] + np.logaddexp(
                alpha[t - 1] + log_self_transition, advance
            )
        likelihood = float(logsumexp(alpha[-1]))
        if not need_expectations:
            return likelihood, None, None

        beta = np.zeros_like(emissions)
        for t in range(len(sequence) - 2, -1, -1):
            advance = np.full(self.n_components, -np.inf)
            advance[:-1] = (
                log_next_transition + emissions[t + 1, 1:] + beta[t + 1, 1:]
            )
            beta[t] = np.logaddexp(
                log_self_transition + emissions[t + 1] + beta[t + 1],
                advance,
            )
        gamma = np.exp(np.clip(alpha + beta - likelihood, -745.0, 0.0))
        log_xi = (
            alpha[:-1, :, None]
            + log_transition
            + emissions[1:, None, :]
            + beta[1:, None, :]
            - likelihood
        )
        transition_counts = np.exp(np.clip(log_xi, -745.0, 0.0)).sum(axis=0)
        return likelihood, gamma, transition_counts

    def fit(self, sequences: list[np.ndarray]) -> "GaussianHMM":
        if not sequences or any(sequence.ndim != 2 for sequence in sequences):
            raise ValueError("fit expects one or more 2-D feature sequences")
        if any(len(sequence) < self.n_components for sequence in sequences):
            raise ValueError("Each training sequence must have at least one frame per state")
        dimensions = sequences[0].shape[1]
        if any(sequence.shape[1] != dimensions for sequence in sequences):
            raise ValueError("All sequences must have the same feature dimension")

        state_frames: list[list[np.ndarray]] = [[] for _ in range(self.n_components)]
        for sequence in sequences:
            boundaries = np.linspace(0, len(sequence), self.n_components + 1, dtype=int)
            for state in range(self.n_components):
                state_frames[state].append(sequence[boundaries[state] : boundaries[state + 1]])
        self.means_ = np.stack(
            [np.vstack(frames).mean(axis=0) for frames in state_frames]
        )
        if self.covariance_type == "diag":
            self.covars_ = np.stack(
                [
                    np.maximum(np.vstack(frames).var(axis=0), self.min_covar)
                    for frames in state_frames
                ]
            )
        else:
            self.covars_ = np.stack(
                [
                    np.atleast_2d(np.cov(np.vstack(frames), rowvar=False))
                    + np.eye(dimensions) * self.min_covar
                    for frames in state_frames
                ]
            )

        self.monitor_ = ConvergenceMonitor()
        for iteration in range(self.n_iter):
            total_likelihood = 0.0
            transition_counts = np.zeros_like(self.transmat_)
            state_weights = np.zeros(self.n_components)
            first_moments = np.zeros_like(self.means_)
            second_moments = (
                np.zeros_like(self.covars_)
                if self.covariance_type == "diag"
                else np.zeros_like(self.covars_)
            )
            for sequence in sequences:
                likelihood, gamma, counts = self._forward_backward(
                    sequence, need_expectations=True
                )
                if gamma is None or counts is None:
                    raise RuntimeError("Forward-backward did not return EM statistics")
                total_likelihood += likelihood
                transition_counts += counts
                state_weights += gamma.sum(axis=0)
                first_moments += gamma.T @ sequence
                if self.covariance_type == "diag":
                    second_moments += gamma.T @ (sequence * sequence)
                else:
                    for state in range(self.n_components):
                        second_moments[state] += (
                            sequence.T * gamma[:, state]
                        ) @ sequence

            self.monitor_.history.append(total_likelihood)
            self.monitor_.iter = iteration + 1
            for state in range(self.n_components):
                if state_weights[state] > 1e-8:
                    self.means_[state] = first_moments[state] / state_weights[state]
            if self.covariance_type == "diag":
                for state in range(self.n_components):
                    if state_weights[state] > 1e-8:
                        variance = (
                            second_moments[state] / state_weights[state]
                            - self.means_[state] ** 2
                        )
                        self.covars_[state] = np.maximum(variance, self.min_covar)
            else:
                for state in range(self.n_components):
                    if state_weights[state] > 1e-8:
                        covariance = (
                            second_moments[state] / state_weights[state]
                            - np.outer(self.means_[state], self.means_[state])
                        )
                        self.covars_[state] = (
                            covariance + covariance.T
                        ) / 2.0 + np.eye(dimensions) * self.min_covar

            transition_counts[~self.allowed_transitions_] = 0.0
            transition_counts[self.allowed_transitions_] += 1e-8
            row_sums = transition_counts.sum(axis=1, keepdims=True)
            self.transmat_ = np.divide(
                transition_counts,
                row_sums,
                out=self.transmat_.copy(),
                where=row_sums > 0,
            )
            if len(self.monitor_.history) > 1:
                improvement = self.monitor_.history[-1] - self.monitor_.history[-2]
                if abs(improvement) < self.tol:
                    self.monitor_.converged = True
                    break
        return self

    def score(self, sequence: np.ndarray) -> float:
        return self._forward_backward(sequence)[0]

    def decode(self, sequence: np.ndarray, algorithm: str = "viterbi") -> tuple[float, np.ndarray]:
        if algorithm != "viterbi":
            raise ValueError("Only Viterbi decoding is supported")
        emissions = self._log_emissions(sequence)
        log_transition = np.full_like(self.transmat_, -np.inf)
        positive = self.transmat_ > 0
        log_transition[positive] = np.log(self.transmat_[positive])
        log_start = np.full(self.n_components, -np.inf)
        positive_start = self.startprob_ > 0
        log_start[positive_start] = np.log(self.startprob_[positive_start])
        delta = np.empty_like(emissions)
        backpointer = np.zeros((len(sequence), self.n_components), dtype=np.int64)
        delta[0] = log_start + emissions[0]
        for t in range(1, len(sequence)):
            candidates = delta[t - 1, :, None] + log_transition
            backpointer[t] = np.argmax(candidates, axis=0)
            delta[t] = emissions[t] + np.max(candidates, axis=0)
        states = np.empty(len(sequence), dtype=np.int64)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(len(sequence) - 1, 0, -1):
            states[t - 1] = backpointer[t, states[t]]
        return float(np.max(delta[-1])), states


def make_lr_hmm(
    n_states: int = 5, covariance_type: str = "diag", random_state: int = 0
) -> GaussianHMM:
    del random_state
    return GaussianHMM(n_components=n_states, covariance_type=covariance_type)


def train_models(
    train: dict[str, list[Path]],
    n_states: int = 5,
    dynamic: bool = True,
    covariance_type: str = "diag",
) -> tuple[dict[str, GaussianHMM], dict[str, list[np.ndarray]], dict[str, float]]:
    models: dict[str, GaussianHMM] = {}
    features: dict[str, list[np.ndarray]] = {}
    seconds: dict[str, float] = {}
    dimension = FEATURE_DIM if dynamic else N_MFCC
    for label in LABELS:
        paths = train[label]
        if not paths:
            raise ValueError(f"No training WAVs found for '{label}'")
        sequences = [extract_features(path, dynamic=dynamic) for path in paths]
        if any(sequence.shape[1] != dimension for sequence in sequences):
            raise RuntimeError(f"Expected {dimension}-D features for '{label}'")
        if any(len(sequence) < n_states for sequence in sequences):
            raise ValueError(
                f"'{label}' contains an utterance shorter than {n_states} frames; "
                "check the audio or reduce the number of states."
            )
        model = make_lr_hmm(n_states, covariance_type)
        started = time.perf_counter()
        model.fit(sequences)
        seconds[label] = time.perf_counter() - started
        if not np.isfinite(model.means_).all() or not np.isfinite(model.covars_).all():
            raise RuntimeError(f"HMM training produced invalid parameters for '{label}'")
        if np.any(model.transmat_[~model.allowed_transitions_] != 0):
            raise RuntimeError(f"Left-to-right topology was not preserved for '{label}'")
        models[label] = model
        features[label] = sequences
        monitor = model.monitor_
        print(
            f"{label}: {len(paths)} train, {n_states} states, "
            f"converged={monitor.converged}, iterations={monitor.iter}, "
            f"last_log_likelihood={monitor.history[-1]:.2f}",
            flush=True,
        )
        print(np.round(model.transmat_, 3), flush=True)
    return models, features, seconds


def score_features(
    features: np.ndarray, models: dict[str, GaussianHMM]
) -> dict[str, float]:
    return {label: float(model.score(features)) for label, model in models.items()}


def predict(
    features: np.ndarray, models: dict[str, GaussianHMM]
) -> tuple[str, dict[str, float]]:
    scores = score_features(features, models)
    return max(scores, key=scores.get), scores


def dtw_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return the minimum normalized Euclidean DTW path cost."""
    return float(dtw_distances(first, [second])[0])


def dtw_distances(first: np.ndarray, references: list[np.ndarray]) -> np.ndarray:
    """Compute exact DTW costs for multiple templates in a batched DP."""
    if first.ndim != 2 or not references or any(
        reference.ndim != 2 or len(reference) == 0 for reference in references
    ):
        raise ValueError("DTW requires non-empty, 2-D feature sequences")
    if len(first) == 0 or any(
        reference.shape[1] != first.shape[1] for reference in references
    ):
        raise ValueError("DTW feature sequences must have matching dimensions")
    rows = len(first)
    reference_lengths = np.asarray([len(reference) for reference in references])
    max_columns = int(reference_lengths.max())
    batch_size = len(references)
    local = np.full((batch_size, rows, max_columns), np.inf, dtype=np.float64)
    for index, reference in enumerate(references):
        local[index, :, : len(reference)] = cdist(
            first, reference, metric="euclidean"
        )
    costs = np.full_like(local, np.inf)
    lengths = np.zeros((batch_size, rows, max_columns), dtype=np.int32)
    costs[:, 0, 0] = local[:, 0, 0]
    lengths[:, 0, 0] = 1
    batch_indices = np.arange(batch_size)[:, None]
    for diagonal in range(1, rows + max_columns - 1):
        start = max(0, diagonal - max_columns + 1)
        stop = min(rows - 1, diagonal)
        i = np.arange(start, stop + 1)
        j = diagonal - i
        candidates = np.full(
            (batch_size, 3, len(i)), np.inf, dtype=np.float64
        )
        if np.any((i > 0) & (j > 0)):
            mask = (i > 0) & (j > 0)
            candidates[:, 0, mask] = costs[
                :, i[mask] - 1, j[mask] - 1
            ]
        if np.any(i > 0):
            mask = i > 0
            candidates[:, 1, mask] = costs[:, i[mask] - 1, j[mask]]
        if np.any(j > 0):
            mask = j > 0
            candidates[:, 2, mask] = costs[:, i[mask], j[mask] - 1]

        best = np.argmin(candidates, axis=1)
        best_cost = np.take_along_axis(candidates, best[:, None, :], axis=1)[:, 0]
        active = j[None, :] < reference_lengths[:, None]
        current_cost = local[batch_indices, i[None, :], j[None, :]]
        costs[batch_indices, i[None, :], j[None, :]] = np.where(
            active, current_cost + best_cost, np.inf
        )

        predecessor_lengths = np.zeros((batch_size, len(i)), dtype=np.int32)
        for direction, (offset_i, offset_j) in enumerate(
            ((-1, -1), (-1, 0), (0, -1))
        ):
            selected = (best == direction) & active
            if not np.any(selected):
                continue
            shifted_i = np.clip(i + offset_i, 0, rows - 1)
            shifted_j = np.clip(j + offset_j, 0, max_columns - 1)
            previous_lengths = lengths[
                batch_indices, shifted_i[None, :], shifted_j[None, :]
            ]
            predecessor_lengths[selected] = previous_lengths[selected]
        lengths[batch_indices, i[None, :], j[None, :]] = np.where(
            active, predecessor_lengths + 1, 0
        )

    final_columns = reference_lengths - 1
    final_costs = costs[
        np.arange(batch_size), rows - 1, final_columns
    ]
    final_lengths = lengths[
        np.arange(batch_size), rows - 1, final_columns
    ]
    return final_costs / np.maximum(final_lengths, 1)


def bigram_counts(
    corpus_path: Path,
) -> tuple[dict[tuple[str, str], int], dict[str, int], dict[str, int], set[str]]:
    pairs: dict[tuple[str, str], int] = {}
    previous_counts: dict[str, int] = {}
    unigram_counts: dict[str, int] = {}
    vocabulary: set[str] = set(LABELS)
    with corpus_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            sequence = []
            for token in line.split():
                label = canonical_label(token)
                if label is None:
                    raise ValueError(
                        f"Unknown word '{token}' in {corpus_path}:{line_number}"
                    )
                sequence.append(label)
                vocabulary.add(label)
                unigram_counts[label] = unigram_counts.get(label, 0) + 1
            for previous, current in zip(sequence, sequence[1:]):
                pairs[(previous, current)] = pairs.get((previous, current), 0) + 1
                previous_counts[previous] = previous_counts.get(previous, 0) + 1
    return pairs, previous_counts, unigram_counts, vocabulary


def parse_lexicon(path: Path) -> dict[str, tuple[str, ...]]:
    """Parse WORD PHONE... lines without assuming an instructor-specific phone set."""
    entries: dict[str, tuple[str, ...]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError(f"Expected WORD and at least one phone at {path}:{line_number}")
            word, *phones = fields
            key = word.casefold()
            if key in entries:
                raise ValueError(f"Duplicate lexicon entry '{word}' at {path}:{line_number}")
            entries[key] = tuple(phones)
    return entries


def bigram_probability(
    previous: str,
    current: str,
    pairs: dict[tuple[str, str], int],
    previous_counts: dict[str, int],
    vocabulary_size: int,
    k: float = 0.5,
) -> float:
    return (pairs.get((previous, current), 0) + k) / (
        previous_counts.get(previous, 0) + k * vocabulary_size
    )


def plot_features(
    examples: list[tuple[str, np.ndarray]], output_path: Path
) -> None:
    examples = examples[:2]
    if not examples:
        return
    fig, axes = plt.subplots(len(examples), 3, figsize=(13, 3.3 * len(examples)))
    axes = np.atleast_2d(axes)
    for row, (label, features) in enumerate(examples):
        for column, (start, end, title) in enumerate(
            ((0, 13, "MFCC"), (13, 26, "Delta"), (26, 39, "Delta-delta"))
        ):
            image = axes[row, column].imshow(
                features[:, start:end].T, aspect="auto", origin="lower", cmap="viridis"
            )
            axes[row, column].set(title=f"{label}: {title}", xlabel="Frame", ylabel="Coefficient")
            fig.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_viterbi(
    models: dict[str, GaussianHMM],
    examples: list[tuple[str, Path, np.ndarray]],
    output_dir: Path,
    is_test: bool,
) -> None:
    for index, (label, path, features) in enumerate(examples[:2], 1):
        log_likelihood, states = models[label].decode(features, algorithm="viterbi")
        fig, axis = plt.subplots(figsize=(10, 3))
        axis.step(np.arange(len(states)), states, where="post")
        axis.set(
            xlabel="Frame",
            ylabel="State",
            yticks=range(models[label].n_components),
            title=f"Viterbi: {label} — {path.name} ({'test' if is_test else 'training demo'})",
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"viterbi_{index}.png", dpi=150)
        plt.close(fig)
        print(f"Viterbi {path.name}: logp={log_likelihood:.2f}, frames/state="
              f"{np.bincount(states, minlength=models[label].n_components).tolist()}")


def evaluate(
    models: dict[str, GaussianHMM],
    train_features: dict[str, list[np.ndarray]],
    test: dict[str, list[Path]],
    output_dir: Path,
    corpus_path: Path | None,
    previous_word: str | None,
) -> dict[str, float] | None:
    rows = []
    examples = []
    y_true: list[str] = []
    y_pred: list[str] = []
    y_dtw: list[str] = []
    incorrect_class_scores: list[float] = []
    hmm_total_seconds = 0.0
    dtw_total_seconds = 0.0
    templates = {
        label: train_features[label]
        for label in LABELS
    }
    pair_counts: dict[tuple[str, str], int] = {}
    previous_counts: dict[str, int] = {}
    unigram_word_counts: dict[str, int] = {}
    vocabulary = set(LABELS)
    if corpus_path and corpus_path.exists():
        pair_counts, previous_counts, unigram_word_counts, vocabulary = bigram_counts(
            corpus_path
        )
        print(f"Corpus unigram counts: {unigram_word_counts}")
        print(f"Corpus bigram counts: {pair_counts}")
        if not pair_counts:
            warnings.warn(f"No bigram sequences found in {corpus_path}; LM reranking skipped")
    elif corpus_path:
        warnings.warn(f"Corpus file not found: {corpus_path}; LM reranking skipped")

    for actual in LABELS:
        for path in sorted(test[actual]):
            features = extract_features(path)
            started = time.perf_counter()
            prediction, scores = predict(features, models)
            hmm_elapsed = time.perf_counter() - started
            hmm_total_seconds += hmm_elapsed
            started = time.perf_counter()
            all_templates = [
                template for label in LABELS for template in templates[label]
            ]
            all_dtw_distances = dtw_distances(features, all_templates)
            dtw_scores = {}
            offset = 0
            for label in LABELS:
                next_offset = offset + len(templates[label])
                dtw_scores[label] = float(
                    all_dtw_distances[offset:next_offset].min()
                )
                offset = next_offset
            dtw_elapsed = time.perf_counter() - started
            dtw_total_seconds += dtw_elapsed
            row = {
                "file": str(path),
                "reference": actual,
                "prediction": prediction,
                "correct": prediction == actual,
                "top1_top2_margin": sorted(scores.values(), reverse=True)[0]
                - sorted(scores.values(), reverse=True)[1],
                "top_score_per_frame": max(scores.values()) / len(features),
                "true_score_per_frame": scores[actual] / len(features),
                "hmm_seconds": hmm_elapsed,
                "dtw_seconds": dtw_elapsed,
            }
            row.update({f"score_{label}": scores[label] for label in LABELS})
            rows.append(row)
            y_true.append(actual)
            y_pred.append(prediction)
            y_dtw.append(min(dtw_scores, key=dtw_scores.get))
            incorrect_class_scores.extend(
                scores[label] / len(features) for label in LABELS if label != actual
            )
            if len(examples) < 2:
                examples.append((actual, path, features))
            print(
                f"test {path.name}: true={actual}, predicted={prediction}, "
                + ", ".join(f"{label}={scores[label]:.2f}" for label in LABELS)
            )
            if previous_word and pair_counts:
                for weight in (0.0, 0.1, 0.3, 1.0):
                    reranked = {
                        label: scores[label] / len(features)
                        + weight
                        * math.log(
                            bigram_probability(
                                previous_word,
                                label,
                                pair_counts,
                                previous_counts,
                                len(vocabulary),
                            )
                        )
                        for label in LABELS
                    }
                    print(f"  LM lambda={weight:.1f}: {max(reranked, key=reranked.get)}")

    if not rows:
        print("No held-out test WAVs found; accuracy/confusion/DTW test metrics are not reported.")
        return None

    with (output_dir / "test_scores.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for actual, predicted_label in zip(y_true, y_pred):
        confusion[LABELS.index(actual), LABELS.index(predicted_label)] += 1
    dtw_confusion = np.zeros_like(confusion)
    for actual, predicted_label in zip(y_true, y_dtw):
        dtw_confusion[LABELS.index(actual), LABELS.index(predicted_label)] += 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, matrix, title in zip(axes, (confusion, dtw_confusion), ("HMM", "DTW baseline")):
        image = axis.imshow(matrix, cmap="Blues")
        axis.set(
            title=title,
            xlabel="Predicted",
            ylabel="Reference",
            xticks=range(len(LABELS)),
            yticks=range(len(LABELS)),
            xticklabels=LABELS,
            yticklabels=LABELS,
        )
        axis.tick_params(axis="x", rotation=35)
        for row in range(len(LABELS)):
            for col in range(len(LABELS)):
                axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
        fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)
    hmm_accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    dtw_accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_dtw)))
    print(f"HMM test accuracy: {hmm_accuracy:.1%} ({sum(a == b for a, b in zip(y_true, y_pred))}/{len(y_true)})")
    print(f"DTW test accuracy: {dtw_accuracy:.1%}")
    print(
        f"Mean test scoring time/trial: HMM={hmm_total_seconds / len(rows):.4f}s, "
        f"DTW={dtw_total_seconds / len(rows):.4f}s"
    )
    true_scores_per_frame = np.array(
        [row["true_score_per_frame"] for row in rows], dtype=float
    )
    best_scores_per_frame = np.array(
        [row["top_score_per_frame"] for row in rows], dtype=float
    )
    print(
        "Correct-class log-likelihood/frame distribution: "
        f"mean={true_scores_per_frame.mean():.3f}, "
        f"std={true_scores_per_frame.std():.3f}, "
        f"min={true_scores_per_frame.min():.3f}, max={true_scores_per_frame.max():.3f}"
    )
    print(
        "Best-class log-likelihood/frame distribution: "
        f"mean={best_scores_per_frame.mean():.3f}, "
        f"std={best_scores_per_frame.std():.3f}"
    )
    wrong_scores_per_frame = np.asarray(incorrect_class_scores, dtype=float)
    print(
        "Wrong-class log-likelihood/frame distribution: "
        f"mean={wrong_scores_per_frame.mean():.3f}, "
        f"std={wrong_scores_per_frame.std():.3f}, "
        f"min={wrong_scores_per_frame.min():.3f}, "
        f"max={wrong_scores_per_frame.max():.3f}"
    )
    print(
        "Unknown/reject threshold cannot be calibrated without labelled unknown "
        "utterances; use a separate validation set and inspect score/frame and "
        "top-1/top-2 margin distributions."
    )
    print("HMM confusion matrix (rows=reference, columns=prediction):\n", confusion)
    print("DTW confusion matrix (rows=reference, columns=prediction):\n", dtw_confusion)
    plot_viterbi(models, examples, output_dir, is_test=True)

    # A reproducible synthetic 20 dB SNR mismatch experiment on held-out utterances.
    rng = np.random.default_rng(0)
    noisy_correct = 0
    for actual in LABELS:
        for path in sorted(test[actual]):
            audio = read_audio(path)
            noise_power = np.mean(audio**2) / (10 ** (20 / 10))
            noisy = audio + rng.normal(0.0, np.sqrt(noise_power), size=audio.shape)
            prediction, _ = predict(extract_features_from_audio(noisy), models)
            noisy_correct += prediction == actual
    noisy_accuracy = noisy_correct / len(rows)
    print(f"Synthetic 20 dB SNR HMM accuracy: {noisy_accuracy:.1%} (not a real speaker/mic mismatch)")
    return {"hmm_accuracy": hmm_accuracy, "dtw_accuracy": dtw_accuracy}


def write_model_file(path: Path, models: dict[str, GaussianHMM], n_states: int) -> None:
    with path.open("wb") as stream:
        pickle.dump(
            {
                "labels": LABELS,
                "sample_rate": SAMPLE_RATE,
                "feature_dim": FEATURE_DIM,
                "n_states": n_states,
                "models": models,
            },
            stream,
        )


def run_model_experiments(
    train: dict[str, list[Path]],
    test: dict[str, list[Path]],
    output_path: Path,
    base_models: dict[str, GaussianHMM],
    base_states: int,
    full_covariance: bool = False,
) -> None:
    """Compare requested state counts and static/dynamic features on held-out data."""
    rows: list[dict[str, object]] = []
    configs = [
        (state_count, True, "diag", base_models if state_count == base_states else None)
        for state_count in (3, 5, 7)
    ]
    configs.append((base_states, False, "diag", None))
    if full_covariance and all(len(train[label]) >= 6 for label in LABELS):
        configs.append((base_states, True, "full", None))
    for state_count, dynamic, covariance_type, existing_models in configs:
        description = (
            f"states={state_count}, features={'39-D' if dynamic else '13-D'}, "
            f"covariance={covariance_type}"
        )
        try:
            if existing_models is None:
                models, _, fit_times = train_models(
                    train, state_count, dynamic=dynamic, covariance_type=covariance_type
                )
            else:
                models = existing_models
                fit_times = {}
            correct = 0
            total = 0
            for label in LABELS:
                for path in sorted(test[label]):
                    predicted, _ = predict(
                        extract_features(path, dynamic=dynamic), models
                    )
                    correct += predicted == label
                    total += 1
            accuracy = correct / total
            row = {
                "configuration": description,
                "accuracy": accuracy,
                "train_seconds": sum(fit_times.values()),
                "converged": all(model.monitor_.converged for model in models.values()),
                "status": "ok",
            }
            print(f"Experiment {description}: accuracy={accuracy:.1%}")
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            row = {
                "configuration": description,
                "accuracy": "",
                "train_seconds": "",
                "converged": "",
                "status": f"failed: {error}",
            }
            warnings.warn(f"Experiment {description} failed: {error}")
        rows.append(row)
    if full_covariance and any(len(train[label]) < 6 for label in LABELS):
        rows.append(
            {
                "configuration": f"states={base_states}, features=39-D, covariance=full",
                "accuracy": "",
                "train_seconds": "",
                "converged": "",
                "status": "skipped: at least 6 train utterances per class are recommended",
            }
        )
        print(
            "Full-covariance experiment skipped: at least 6 train utterances per "
            "class are recommended."
        )
    with output_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Train/test Vietnamese command HMMs. WAVs may be in dataset/train/<label>/ and dataset/test/<label>/."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--states", type=int, choices=(3, 5, 7), default=5)
    parser.add_argument("--corpus", type=Path, default=Path("corpus.txt"))
    parser.add_argument("--previous-word", type=str)
    parser.add_argument("--lexicon", type=Path, default=Path("lexicon.txt"))
    parser.add_argument(
        "--full-covariance-experiment",
        action="store_true",
        help="Also compare full covariance when each class has at least 6 training utterances.",
    )
    parser.add_argument(
        "--run-experiments",
        action="store_true",
        help="Compare 3/5/7 states and static/dynamic MFCC; adds training time.",
    )
    args = parser.parse_args()

    train_dir = args.data_dir / "train"
    test_dir = args.data_dir / "test"
    train = collect_split(train_dir) if train_dir.exists() else collect_flat_wavs(Path.cwd())
    test = collect_split(test_dir)
    if not any(train.values()):
        print(
            "No labelled WAVs found. Add files under dataset/train/<label>/ (and "
            "dataset/test/<label>/ for evaluation), or put one labelled WAV per class "
            "in the project root.",
            file=sys.stderr,
        )
        return 2
    check_no_split_leakage(train, test)
    absent = [label for label in LABELS if not train[label]]
    if absent:
        print(f"Missing training examples for: {', '.join(absent)}", file=sys.stderr)
        return 2
    train_count = sum(map(len, train.values()))
    test_count = sum(map(len, test.values()))
    if not test_count:
        print(
            f"Only {train_count} labelled recording(s) available; no held-out test split. "
            "The supplied single recordings are used for model training only, so no "
            "test accuracy or confusion matrix will be claimed."
        )
    elif any(len(files) < 2 for files in train.values()):
        print(
            "Warning: some classes have fewer than two training utterances; HMM "
            "parameters and comparisons may be unstable."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        lexicon = parse_lexicon(args.lexicon) if args.lexicon.exists() else {}
        print(f"Lexicon entries loaded: {len(lexicon)}")
        previous_word = canonical_label(args.previous_word) if args.previous_word else None
        if args.previous_word and previous_word is None:
            raise ValueError(
                f"Unknown --previous-word '{args.previous_word}'; use one of {', '.join(LABELS)}"
            )
        models, train_features, train_seconds = train_models(train, args.states)
        print(f"Total HMM training time: {sum(train_seconds.values()):.3f} s")
        write_model_file(args.output_dir / "models.pkl", models, args.states)
        train_correct = sum(
            predict(features, models)[0] == label
            for label in LABELS
            for features in train_features[label]
        )
        print(
            f"In-sample training accuracy (optimistic, not test): "
            f"{train_correct / train_count:.1%} ({train_correct}/{train_count})"
        )
        examples = []
        for label in LABELS:
            for path, features in zip(train[label], train_features[label]):
                examples.append((label, path, features))
        plot_features(
            [(label, features) for label, _, features in examples],
            args.output_dir / "mfcc_heatmaps.png",
        )
        if test_count:
            evaluate(
                models,
                train_features,
                test,
                args.output_dir,
                args.corpus if args.corpus.exists() else None,
                previous_word,
            )
            if args.run_experiments or args.full_covariance_experiment:
                run_model_experiments(
                    train,
                    test,
                    args.output_dir / "experiments.csv",
                    models,
                    args.states,
                    args.full_covariance_experiment,
                )
        else:
            plot_viterbi(
                models,
                [(label, path, features) for label, path, features in examples[:2]],
                args.output_dir,
                is_test=False,
            )
            print(
                "In-sample demonstration scores (NOT a test result): "
                "see the printed model likelihoods above. Supply a disjoint "
                "dataset/test split to produce test_scores.csv and confusion matrices."
            )
    except (ValueError, OSError, RuntimeError, np.linalg.LinAlgError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(f"Outputs saved to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
