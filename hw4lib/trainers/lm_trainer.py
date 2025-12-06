from .base_trainer import BaseTrainer
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Tuple, Any, Optional, List
from ..utils import create_scheduler
from ..decoding.sequence_generator import SequenceGenerator


class LMTrainer(BaseTrainer):
    """
    Language Model Trainer class that handles the training, validation, and generation loops.

    This trainer implements:
    1. Training loop with gradient accumulation and mixed precision training
    2. Validation loop for model evaluation
    3. Generation capabilities with different decoding strategies

    You only need to fill in the TODOs in the code. 
    Please do not modify any other code without understanding what you are doing.
    
    Implementation Tasks:
    - TODO: Initialize the criterion in __init__
    - TODO: Implement key parts of the training loop in _train_epoch
    - TODO: Use your greedy generation implementation in generate
    - TODO: Implement key parts of the the validation loop in _validate_epoch
    - TODO: Implement key parts of the full training loop in train

    Implementation Notes:
    1. For __init__:
        - Initialize CrossEntropyLoss with appropriate padding index and label smoothing
        
    2. For _train_epoch:
        - Unpack the batch (shifted inputs, golden targets, lengths)
        - Get model predictions and attention weights
        - Calculate loss
        
    3. For _validate_epoch:
        - Similar to _train_epoch but without gradient calculations
        - Use torch.inference_mode() for validation
        
    4. For train:
        - Implement the epoch loop with training and validation and generation
        
    5. For generate:
        - Use the greedy decoding method you implemented in SequenceGenerator
        - Post-process sequences using appropriate tokenizer methods
        - Format results
    """

    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)

        # TODO: Initialize the loss criterion
        # Use CrossEntropyLoss with padding token ignored and label smoothing
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=self.config["loss"]["label_smoothing"]
        )

    def _count_characters(self, token_ids, tokenizer):
        """
        Count the actual number of characters represented by the given token IDs.
        
        Args:
            token_ids: Tensor of token IDs [batch_size, seq_len]
            tokenizer: Tokenizer to decode tokens
            
        Returns:
            Total number of characters (excluding padding, SOS, EOS)
        """
        total_chars = 0
        batch_size, seq_len = token_ids.shape
        
        # Special token IDs to exclude
        special_token_ids = {
            tokenizer.pad_id,
            tokenizer.sos_id,
            tokenizer.eos_id,
            tokenizer.unk_id,
            tokenizer.mask_id,
            tokenizer.blank_id
        }
        
        for i in range(batch_size):
            # Get tokens for this sequence
            tokens = token_ids[i].tolist()
            
            # Filter out all special tokens
            tokens = [t for t in tokens if t not in special_token_ids]
            
            # Decode to get actual text and count characters
            if len(tokens) > 0:
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                total_chars += len(text)
        
        return total_chars

    # ======================================================================
    #                           TRAIN EPOCH
    # ======================================================================
    def _train_epoch(self, dataloader):
        """
        Train the model for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Tuple of (metrics_dict, attention_weights)
        """
        self.model.train()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False,
                         desc=f"[Training LM]")

        running_ce_loss = 0.0
        total_tokens = 0
        total_chars = 0
        self.optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            # TODO: Unpack the batch (shifted inputs, golden targets, lengths)
            targets_shifted, targets_golden, lengths = batch
            targets_shifted = targets_shifted.to(self.device)
            targets_golden = targets_golden.to(self.device)
            lengths = lengths.to(self.device)

            with torch.autocast(device_type=str(self.device), dtype=torch.float16):
                # TODO: Get model predictions and attention weights
                raw_preds, attn_weights = self.model(targets_shifted)

                # TODO: Calculate loss
                # The model output shape is [batch_size, seq_len, vocab_size]
                # targets_golden shape is [batch_size, seq_len]
                # No slicing needed - the alignment is already correct from dataset
                raw_loss = self.ce_criterion(
                    raw_preds.reshape(-1, raw_preds.size(-1)),
                    targets_golden.reshape(-1)
                )

            # Calculate both token and character counts
            batch_tokens = lengths.sum().item()
            batch_chars = self._count_characters(targets_golden, self.tokenizer)
            
            running_ce_loss += raw_loss.item() * batch_tokens
            total_tokens += batch_tokens
            total_chars += batch_chars

            # Gradient accumulation
            loss = raw_loss / self.config['training']['gradient_accumulation_steps']
            self.scaler.scale(loss).backward()

            if (i + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.scaler.step(self.optimizer)
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                self.scaler.update()
                self.optimizer.zero_grad()

            # Calculate running metrics
            avg_ce_loss_token = running_ce_loss / total_tokens
            avg_ce_loss_char = running_ce_loss / total_chars if total_chars > 0 else avg_ce_loss_token
            perplexity_token = torch.exp(torch.tensor(avg_ce_loss_token))
            perplexity_char = torch.exp(torch.tensor(avg_ce_loss_char))
            
            batch_bar.set_postfix(
                ce_loss_token=f"{avg_ce_loss_token:.4f}",
                ce_loss_char=f"{avg_ce_loss_char:.4f}",
                perplexity_char=f"{perplexity_char:.4f}"
            )
            batch_bar.update()

        batch_bar.close()

        # Final epoch metrics
        avg_ce_loss_token = running_ce_loss / total_tokens
        avg_ce_loss_char = running_ce_loss / total_chars if total_chars > 0 else avg_ce_loss_token
        perplex_token = torch.exp(torch.tensor(avg_ce_loss_token)).item()
        perplex_char = torch.exp(torch.tensor(avg_ce_loss_char)).item()

        return {
            'ce_loss_token': avg_ce_loss_token,
            'ce_loss_char': avg_ce_loss_char,
            'perplexity_token': perplex_token,
            'perplexity_char': perplex_char
        }, attn_weights

    # ======================================================================
    #                           VALIDATION EPOCH
    # ======================================================================
    def _validate_epoch(self, dataloader):
        """
        Validate the model for one epoch.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Tuple of (metrics_dict, attention_weights)
        """
        self.model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False,
                         desc=f"[Validating LM]")

        running_ce_loss = 0.0
        total_tokens = 0
        total_chars = 0

        for batch in dataloader:
            # TODO: Unpack the batch (shifted inputs, golden targets, lengths)
            targets_shifted, targets_golden, lengths = batch
            targets_shifted = targets_shifted.to(self.device)
            targets_golden = targets_golden.to(self.device)
            lengths = lengths.to(self.device)

            with torch.inference_mode():
                # TODO: Get model predictions and attention weights
                raw_preds, attn_weights = self.model(targets_shifted)

                # TODO: Calculate loss
                # No slicing needed - alignment is correct from dataset
                loss = self.ce_criterion(
                    raw_preds.reshape(-1, raw_preds.size(-1)),
                    targets_golden.reshape(-1)
                )

            # Calculate both token and character counts
            batch_tokens = lengths.sum().item()
            batch_chars = self._count_characters(targets_golden, self.tokenizer)
            
            running_ce_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            total_chars += batch_chars

            # Calculate running metrics
            avg_ce_loss_token = running_ce_loss / total_tokens
            avg_ce_loss_char = running_ce_loss / total_chars if total_chars > 0 else avg_ce_loss_token
            perplexity_token = torch.exp(torch.tensor(avg_ce_loss_token))
            perplexity_char = torch.exp(torch.tensor(avg_ce_loss_char))

            batch_bar.set_postfix(
                ce_loss_token=f"{avg_ce_loss_token:.4f}",
                ce_loss_char=f"{avg_ce_loss_char:.4f}",
                perplexity_char=f"{perplexity_char:.4f}"
            )
            batch_bar.update()

        batch_bar.close()

        # Final epoch metrics
        avg_ce_loss_token = running_ce_loss / total_tokens
        avg_ce_loss_char = running_ce_loss / total_chars if total_chars > 0 else avg_ce_loss_token
        perplex_token = torch.exp(torch.tensor(avg_ce_loss_token)).item()
        perplex_char = torch.exp(torch.tensor(avg_ce_loss_char)).item()

        return {
            'ce_loss_token': avg_ce_loss_token,
            'ce_loss_char': avg_ce_loss_char,
            'perplexity_token': perplex_token,
            'perplexity_char': perplex_char
        }, attn_weights

    # ======================================================================
    #                           FULL TRAIN LOOP
    # ======================================================================
    def train(self, train_dataloader, val_dataloader, epochs: int):
        """
        Full training loop with training, validation, and generation.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            epochs: Number of epochs to train
        """
        # TODO: Implement the epoch loop with training, validation and generation
        if self.scheduler is None:
            raise ValueError("Scheduler not initialized!")
        if self.optimizer is None:
            raise ValueError("Optimizer not initialized!")

        best_val_loss = float('inf')

        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            # Training phase
            train_metrics, train_attn = self._train_epoch(train_dataloader)
            
            # Validation phase
            val_metrics, val_attn = self._validate_epoch(val_dataloader)
            
            # Generation phase
            gen_results = self.generate(val_dataloader)

            # Update learning rate scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['ce_loss_char'])

            # Logging
            metrics = {'train': train_metrics, 'val': val_metrics}
            self._log_metrics(metrics, epoch)

            # Save attention plots
            self._save_attention_plot(train_attn[list(train_attn.keys())[0]][0], epoch, "train_self")
            self._save_attention_plot(val_attn[list(val_attn.keys())[0]][0], epoch, "val_self")

            # Save generated sequences
            self._save_generated_text(gen_results, f'val_epoch_{epoch}')

            # Save checkpoint
            self.save_checkpoint('checkpoint-last-epoch-model.pth')

            # Save best model based on validation loss
            if val_metrics['ce_loss_char'] < best_val_loss:
                best_val_loss = val_metrics['ce_loss_char']
                self.best_metric = best_val_loss
                self.save_checkpoint('checkpoint-best-metric-model.pth')

            self.current_epoch += 1

    # ======================================================================
    #                           GENERATION
    # ======================================================================
    def evaluate(self, test_dataloader):
        """
        Evaluate the model on the test set.
        
        Args:
            test_dataloader: DataLoader for test data
        Returns:
            Tuple[Dict[str, float], Dict[str, Dict[str, Dict]]]: A tuple containing:
                - test_metrics: Test metrics
                - generation_results: Generation results for each config
        """
        test_metrics, test_attn = self._validate_epoch(test_dataloader)

        # Log metrics
        metrics = {
            'test': test_metrics
        }
        self._log_metrics(metrics, self.current_epoch)  

        # Save attention plots
        test_attn_keys = list(test_attn.keys())
        self._save_attention_plot(test_attn[test_attn_keys[0]][0], self.current_epoch, "test_self")

        # Generate with evaluation configs and collect results
        generation_results = {}
        eval_configs = self._get_evaluation_generation_configs()
        for config_name, config in eval_configs.items():
            try:
                gen_results = self.generate(test_dataloader, generation_config=config)
                generation_results[config_name] = gen_results
                self._save_generated_text(gen_results, f'test_epoch_{self.current_epoch}_{config_name}')
            except Exception as e:
                print(f"Could not generate results for {config_name}: {e}")
                continue
        return test_metrics, generation_results
    def generate(self, dataloader, generation_config=None):
        """
        Generate text using greedy decoding.
        
        Args:
            dataloader: Data loader to sample prompts from
            generation_config: Optional configuration for generation
            
        Returns:
            List of dictionaries containing generation results
        """
        # Default generation settings
        if generation_config is None:
            generation_config = {
                "num_samples": 10,
                "prompt_length": 20,
                "seed": 11785,
                "temperature": 1.0,
                "beam_width": 1,
                "repeat_penalty": 1.0,
            }

        # TODO: Build generator
        generator = SequenceGenerator(
            score_fn=lambda x: self.model.score(x),
            tokenizer=self.tokenizer,
            max_length=self.model.max_len,
            device=self.device,
        )

        # Use dataset's sample_prompts()
        prompts, originals = dataloader.dataset.sample_prompts(
            num_samples=generation_config["num_samples"],
            prompt_length=generation_config["prompt_length"],
            seed=generation_config["seed"],
        )

        prompts = prompts.to(self.device)

        # TODO: Run greedy decoding using your implementation
        self.model.eval()
        with torch.inference_mode():
            seqs, scores = generator.generate_greedy(
                prompts,
                temperature=generation_config["temperature"],
                repeat_penalty=generation_config["repeat_penalty"],
            )

        # Post-process sequences
        processed = generator.post_process_sequence(seqs, self.tokenizer)

        # Safely reconstruct output text
        results = []
        for p, seq, s, orig in zip(prompts, processed, scores, originals):
            p_list = p.tolist()
            seq_list = seq.tolist()
            orig_list = orig.tolist()

            # Strip padding tokens
            pad = self.tokenizer.pad_id
            p_clean = [t for t in p_list if t != pad]
            orig_clean = [t for t in orig_list if t != pad]

            # Ensure slicing range is valid to avoid index errors
            prompt_len = len(p_clean)
            if prompt_len < len(seq_list):
                gen_only = seq_list[prompt_len:]
            else:
                gen_only = []

            # Also ensure original has enough tokens
            if prompt_len < len(orig_clean):
                orig_continuation = orig_clean[prompt_len:]
            else:
                orig_continuation = []

            results.append({
                "prompt": self.tokenizer.decode(p_clean),
                "original": self.tokenizer.decode(orig_continuation),
                "generated": self.tokenizer.decode(gen_only),
                "score": float(s.item()),
            })

        return results

    def _get_evaluation_generation_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a list of generation configurations for evaluation.
        
        Returns:
            Dictionary containing generation configurations
        """
        common_config = {
            'num_samples': 50,
            'prompt_length': 10,
            'seed': 11785,
            'max_length': self.model.max_len,
        }
        
        greedy_config = common_config.copy()
        greedy_config.update({
            'temperature': 1.0,
            'beam_width': 1,
            'repeat_penalty': 1.0,
            'top_k': 0,
            'top_p': 0.0
        })
        
        beam_config = common_config.copy()
        beam_config.update({
            'temperature': 1.0,
            'beam_width': 10,
            'repeat_penalty': 1.2,
            'top_k': 0,
            'top_p': 0.0
        })

        sample_config = common_config.copy()
        sample_config.update({
            'temperature': 1.0,
            'beam_width': 1,
            'repeat_penalty': 1.0,
            'top_k': 10,
            'top_p': 0.95
        })
        
        return {
            'greedy': greedy_config,
            'beam': beam_config,
            'sample': sample_config
        }