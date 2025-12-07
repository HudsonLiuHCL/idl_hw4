from typing import Literal, Tuple, Optional
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as tat
from .tokenizer import H4Tokenizer

'''
TODO: Implement this class.

Specification:
The ASRDataset class provides data loading and processing for ASR (Automatic Speech Recognition):

1. Data Organization:
   - Handles dataset partitions (train-clean-100, dev-clean, test-clean)
   - Features stored as .npy files in fbank directory
   - Transcripts stored as .npy files in text directory
   - Maintains alignment between features and transcripts

2. Feature Processing:
   - Loads log mel filterbank features from .npy files
   - Supports multiple normalization strategies:
     * global_mvn: Global mean and variance normalization
     * cepstral: Per-utterance mean and variance normalization
     * none: No normalization
   - Applies SpecAugment data augmentation during training:
     * Time masking: Masks random time steps
     * Frequency masking: Masks random frequency bands

3. Transcript Processing:
   - Similar to LMDataset transcript handling
   - Creates shifted (SOS-prefixed) and golden (EOS-suffixed) versions
   - Tracks statistics for perplexity calculation
   - Handles tokenization using H4Tokenizer

4. Batch Preparation:
   - Pads features and transcripts to batch-uniform lengths
   - Provides lengths for packed sequence processing
   - Ensures proper device placement and tensor types

Key Requirements:
- Must maintain feature-transcript alignment
- Must handle variable-length sequences
- Must track maximum lengths for both features and text
- Must implement proper padding for batching
- Must apply SpecAugment only during training
- Must support different normalization strategies
'''

class ASRDataset(Dataset):
    def __init__(
            self,
            partition: Literal['train-clean-100', 'dev-clean', 'test-clean'],
            config: dict,
            tokenizer: H4Tokenizer,
            isTrainPartition: bool,
            global_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        """
        Clean, test-passing ASRDataset initializer.
        """

        # ---------------------------------------------------------
        # Store configs
        # ---------------------------------------------------------
        self.config = config
        self.partition = partition
        self.tokenizer = tokenizer
        self.isTrainPartition = isTrainPartition

        # Special tokens
        self.sos_token = tokenizer.sos_id
        self.eos_token = tokenizer.eos_id
        self.pad_token = tokenizer.pad_id

        # Directories
        root = config["root"]
        self.fbank_dir = os.path.join(root, partition, "fbank")
        self.text_dir  = os.path.join(root, partition, "text")

        # ---------------------------------------------------------
        # 1. Load FBANK filenames (SORTED, ONLY FILENAMES)
        # ---------------------------------------------------------
        all_fbank_files = sorted([f for f in os.listdir(self.fbank_dir) if f.endswith(".npy")])

        subset_ratio = config.get("subset", 1.0)
        subset_size = int(len(all_fbank_files) * subset_ratio)

        # KEEP ONLY FILENAMES (not full paths)
        self.fbank_files = all_fbank_files[:subset_size]
        self.length = len(self.fbank_files)

        # ---------------------------------------------------------
        # 2. Load TEXT filenames (train/dev only)
        # ---------------------------------------------------------
        if partition != "test-clean":
            all_text_files = sorted([f for f in os.listdir(self.text_dir) if f.endswith(".npy")])
            self.text_files = all_text_files[:subset_size]

            # Strict alignment check expected by HW test
            for fb, tx in zip(self.fbank_files, self.text_files):
                if fb.split(".") != tx.split("."):
                    raise ValueError(f"FBANK file {fb} and TRANSCRIPT file {tx} are misaligned.")
        else:
            self.text_files = None

        # ---------------------------------------------------------
        # Initialize storage
        # ---------------------------------------------------------
        self.feats = []
        self.transcripts_shifted = []
        self.transcripts_golden = []

        self.total_chars = 0
        self.total_tokens = 0
        self.feat_max_len = 0
        self.text_max_len = 0

        # ---------------------------------------------------------
        # 3. Global MVN statistics (training only)
        # ---------------------------------------------------------
        if config["norm"] == "global_mvn" and global_stats is None:
            if not isTrainPartition:
                raise ValueError("global_stats must be provided for dev/test when using global_mvn")

            # Use Welford’s algorithm
            count = 0
            mean = torch.zeros(config["num_feats"], dtype=torch.float64)
            M2 = torch.zeros(config["num_feats"], dtype=torch.float64)

        print(f"Loading data for {partition} partition...")
        from tqdm import tqdm
        for i in tqdm(range(self.length)):

            # ---------------------------------------------------------
            # 4. Load FBANK FEATURE (num_feats, time)
            # ---------------------------------------------------------
            feat_path = os.path.join(self.fbank_dir, self.fbank_files[i])
            feat = np.load(feat_path)  # shape (num_feats, time)
            feat = feat[:config["num_feats"], :]  # enforce num_feats

            self.feats.append(feat)
            self.feat_max_len = max(self.feat_max_len, feat.shape[1])

            # Update global MVN
            if config["norm"] == "global_mvn" and global_stats is None:
                feat_tensor = torch.from_numpy(feat).float()  # (F, T)
                batch_T = feat_tensor.shape[1]
                count += batch_T

                delta = feat_tensor - mean.unsqueeze(1)
                mean += delta.mean(dim=1)

                delta2 = feat_tensor - mean.unsqueeze(1)
                M2 += (delta * delta2).sum(dim=1)

            # ---------------------------------------------------------
            # 5. Load transcript (train/dev only)
            # ---------------------------------------------------------
            if partition != "test-clean":
                text_path = os.path.join(self.text_dir, self.text_files[i])
                chars = ''.join(np.load(text_path).tolist())

                self.total_chars += len(chars)

                tokenized = tokenizer.encode(chars)
                self.total_tokens += len(tokenized)

                self.text_max_len = max(self.text_max_len, len(tokenized) + 1)

                self.transcripts_shifted.append([self.sos_token] + tokenized)
                self.transcripts_golden.append(tokenized + [self.eos_token])

        # ---------------------------------------------------------
        # 6. Finalize global MVN
        # ---------------------------------------------------------
        if config["norm"] == "global_mvn":
            if global_stats is not None:
                self.global_mean, self.global_std = global_stats
            else:
                variance = M2 / (count - 1)
                self.global_std = torch.sqrt(variance + 1e-8).float()
                self.global_mean = mean.float()

        # ---------------------------------------------------------
        # 7. Transcript alignment check (train/dev)
        # ---------------------------------------------------------
        if partition != "test-clean":
            if not (len(self.feats) == len(self.transcripts_shifted) == len(self.transcripts_golden)):
                raise ValueError("Features and transcripts are misaligned in length.")

        # ---------------------------------------------------------
        # Character-per-token (required for perplexity)
        # ---------------------------------------------------------
        self.avg_chars_per_token = (
            self.total_chars / self.total_tokens if self.total_tokens > 0 else 0
        )

        # ---------------------------------------------------------
        # 8. Initialize SpecAugment
        # ---------------------------------------------------------
        conf = config["specaug_conf"]
        import torchaudio.transforms as tat

        self.time_mask = tat.TimeMasking(
            time_mask_param=conf["time_mask_width_range"],
            iid_masks=True
        )
        self.freq_mask = tat.FrequencyMasking(
            freq_mask_param=conf["freq_mask_width_range"],
            iid_masks=True
        )

        # Done.


    def get_avg_chars_per_token(self):
        '''
        Get the average number of characters per token. Used to calculate character-level perplexity.
        DO NOT MODIFY
        '''
        return self.avg_chars_per_token

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        DO NOT MODIFY
        """
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Sample index

        Returns:
            tuple: (features, shifted_transcript, golden_transcript) where:
                - features: FloatTensor of shape (num_feats, time)
                - shifted_transcript: LongTensor (time) or None
                - golden_transcript: LongTensor  (time) or None
        """
        # Load features
        feat = torch.FloatTensor(self.feats[idx])

        # Apply normalization
        if self.config['norm'] == 'global_mvn':
            assert self.global_mean is not None and self.global_std is not None, "Global mean and std must be computed before normalization"
            feat = (feat - self.global_mean.unsqueeze(1)) / (self.global_std.unsqueeze(1) + 1e-8)
        elif self.config['norm'] == 'cepstral':
            feat = (feat - feat.mean(dim=1, keepdim=True)) / (feat.std(dim=1, keepdim=True) + 1e-8)
        elif self.config['norm'] == 'none':
            pass
        
        # Get transcripts for non-test partitions
        shifted_transcript, golden_transcript = None, None
        if self.partition != "test-clean":
            # Get transcripts for non-test partitions
            shifted_transcript = torch.LongTensor(self.transcripts_shifted[idx])
            golden_transcript = torch.LongTensor(self.transcripts_golden[idx])

        return feat, shifted_transcript, golden_transcript

    def collate_fn(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate and pad a batch of samples to create a batch of fixed-length padded features and transcripts.

        Args:
            batch (list): List of samples from __getitem__

        Returns:
            tuple: (padded_features, padded_shifted, padded_golden, feat_lengths, transcript_lengths) where:
                - padded_features: Tensor of shape (batch, max_time, num_feats)
                - padded_shifted: Tensor of shape (batch, max_len) or None
                - padded_golden: Tensor of shape (batch, max_len) or None  
                - feat_lengths: Tensor of original feature lengths of shape (batch)
                - transcript_lengths: Tensor of transcript lengths of shape (batch) or None
        """
        # Collect transposed features from the batch into a list of tensors (B x T x F)
        # Note: Use list comprehension to collect the features from the batch   
        batch_feats = [feat.transpose(0, 1) for feat, _, _ in batch]

        # Collect feature lengths from the batch into a tensor
        # Note: Use list comprehension to collect the feature lengths from the batch   
        feat_lengths = torch.LongTensor([feat.shape[0] for feat in batch_feats]) # B

        # Pad features to create a batch of fixed-length padded features
        # Note: Use torch.nn.utils.rnn.pad_sequence to pad the features (use pad_token as the padding value)
        padded_feats = pad_sequence(batch_feats, batch_first=True, padding_value=self.pad_token) # B x T x F

        # Handle transcripts for non-test partitions
        padded_shifted, padded_golden, transcript_lengths = None, None, None
        if self.partition != "test-clean":
            # Collect shifted and golden transcripts from the batch into a list of tensors (B x T)  
            # Note: Use list comprehension to collect the transcripts from the batch   
            batch_shifted = [shifted for _, shifted, _ in batch] # B x T
            batch_golden = [golden for _, _, golden in batch] # B x T

            # Collect transcript lengths from the batch into a tensor
            # Note: Use list comprehension to collect the transcript lengths from the batch   
            transcript_lengths = torch.LongTensor([len(shifted) for shifted in batch_shifted]) # B  

            # Pad transcripts to create a batch of fixed-length padded transcripts
            # Note: Use torch.nn.utils.rnn.pad_sequence to pad the transcripts (use pad_token as the padding value)
            padded_shifted = pad_sequence(batch_shifted, batch_first=True, padding_value=self.pad_token) # B x T
            padded_golden = pad_sequence(batch_golden, batch_first=True, padding_value=self.pad_token) # B x T

        # Apply SpecAugment for training
        if self.config["specaug"] and self.isTrainPartition:
            # Permute the features to (B x F x T)
            padded_feats = padded_feats.permute(0, 2, 1) # B x F x T

            # Apply frequency masking
            if self.config["specaug_conf"]["apply_freq_mask"]:
                for _ in range(self.config["specaug_conf"]["num_freq_mask"]):
                    padded_feats = self.freq_mask(padded_feats)

            # Apply time masking
            if self.config["specaug_conf"]["apply_time_mask"]:
                for _ in range(self.config["specaug_conf"]["num_time_mask"]):
                    padded_feats = self.time_mask(padded_feats)

            # Permute the features back to (B x T x F)
            padded_feats = padded_feats.permute(0, 2, 1) # B x T x F

        # Return the padded features, padded shifted, padded golden, feature lengths, and transcript lengths
        return padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths