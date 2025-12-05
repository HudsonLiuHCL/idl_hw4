import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
from torch.utils.data import DataLoader
import json
import types

# HW4 libs
from hw4lib.data.tokenizer import H4Tokenizer
from hw4lib.data.lm_dataset import LMDataset
from hw4lib.model.transformers import DecoderOnlyTransformer
from hw4lib.trainers.lm_trainer import LMTrainer
from hw4lib.utils.create_optimizer import create_optimizer
from hw4lib.utils.create_lr_scheduler import create_scheduler


def main():

    ###############################################################
    # TOKENIZER
    ###############################################################
    token_map = {
        "char": "hw4lib/data/tokenizer_jsons/tokenizer_char.json",
        "1k":   "hw4lib/data/tokenizer_jsons/tokenizer_1000.json",
        "5k":   "hw4lib/data/tokenizer_jsons/tokenizer_5000.json",
        "10k":  "hw4lib/data/tokenizer_jsons/tokenizer_10000.json",
    }

    tokenizer = H4Tokenizer(token_map, token_type="char")

    ###############################################################
    # DATASETS
    ###############################################################
    lm_cfg = {"root": "", "subset_size": None}

    train_data = LMDataset("train", lm_cfg, tokenizer)
    val_data   = LMDataset("val",   lm_cfg, tokenizer)
    test_data  = LMDataset("test",  lm_cfg, tokenizer)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True,
                              collate_fn=train_data.collate_fn)
    val_loader   = DataLoader(val_data, batch_size=16, shuffle=False,
                              collate_fn=val_data.collate_fn)
    test_loader  = DataLoader(test_data, batch_size=16, shuffle=False,
                              collate_fn=test_data.collate_fn)

    ###############################################################
    # MODEL  → STRONGER MODEL FOR BETTER PERPLEXITY
    ###############################################################
    model = DecoderOnlyTransformer(
        num_layers=8,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        dropout=0.1,
        max_len=512,
        num_classes=tokenizer.vocab_size,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    ###############################################################
    # OPTIMIZER + SCHEDULER
    ###############################################################
    opt_cfg = {
        "name": "adamw",
        "lr": 2e-4,
        "weight_decay": 0.01,

        "param_groups": [
            {"name": "no_decay", "lr": 2e-4,
             "patterns": ["bias", "LayerNorm.weight"]},
            {"name": "decay", "lr": 2e-4,
             "patterns": [".*"]},
        ],

        "adamw": {
            "betas": (0.9, 0.98),
            "eps": 1e-8,
            "amsgrad": False,
        }
    }

    optimizer = create_optimizer(model, opt_cfg)

    scheduler_cfg = {
        "name": "cosine_warm",
        "cosine_warm": {"T_0": 20, "T_mult": 2, "eta_min": 5e-6},
        "warmup": {"enabled": True, "epochs": 10,
                   "start_factor": 0.1, "end_factor": 1.0},
    }

    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_config=scheduler_cfg,
        train_loader=train_loader,
        gradient_accumulation_steps=1,
    )

    ###############################################################
    # TRAINER CONFIG (LABEL SMOOTHING HELPS PERPLEXITY)
    ###############################################################
    trainer_config = {
        "loss": {"label_smoothing": 0.1},
        "training": {
            "gradient_accumulation_steps": 1,
            "use_wandb": False,
        },
        "data": {"batch_size": 16}
    }

    ###############################################################
    # INITIALIZE TRAINER
    ###############################################################
    trainer = LMTrainer(
        model=model,
        tokenizer=tokenizer,
        config=trainer_config,
        run_name="hw4_experiment_bigmodel",
        config_file="config.yaml",
        device=device
    )

    trainer.optimizer = optimizer
    trainer.scheduler = scheduler

    # Make evaluate() concrete
    def eval_impl(self, dataloader):
        metrics, _ = self._validate_epoch(dataloader)
        return metrics

    trainer.evaluate = types.MethodType(eval_impl, trainer)

    ###############################################################
    # TRAIN (MORE EPOCHS → MUCH LOWER PERPLEXITY)
    ###############################################################
    print("Starting training...")
    trainer.train(train_loader, val_loader, epochs=40)

    ###############################################################
    # VAL EVAL
    ###############################################################
    print("\nFinal Evaluation on Validation Set:")
    val_metrics = trainer.evaluate(val_loader)
    print(val_metrics)

    ###############################################################
    # TEST EVAL
    ###############################################################
    print("\nRunning TEST evaluation...")
    test_metrics = trainer.evaluate(test_loader)

    with open("test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    print("Saved test_metrics.json")

    ###############################################################
    # GENERATION
    ###############################################################
    prompts, originals = test_data.sample_prompts(
        num_samples=1,
        prompt_length=20,
        seed=42
    )
    prompt_ids = prompts.to(device)

    # your trainer doesn't support generate_from_prompt
    gen = trainer.generate(val_loader)

    with open("test_generated_results.json", "w") as f:
        json.dump(gen, f, indent=4)

    print("Saved test_generated_results.json")


if __name__ == "__main__":
    main()
