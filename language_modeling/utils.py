import os
from pathlib import Path
import shutil


def save_checkpoint(model_to_save, tokenizer, global_step, output_dir):
    # delete older checkpoint(s)
    glob_checkpoints = [
        str(x) for x in Path(output_dir).glob(f"checkpoint-*")
    ]

    for checkpoint in glob_checkpoints:
        # logger.info(f"Deleting older checkpoint {checkpoint}")
        shutil.rmtree(checkpoint)

    # Save model checkpoint
    ckpt_output_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(ckpt_output_dir, exist_ok=True)

    # logger.info(f"Saving model checkpoint to {ckpt_output_dir}")
    
    model_to_save.save_pretrained(ckpt_output_dir)
    tokenizer.save_pretrained(ckpt_output_dir)

def dataset_to_text(dataset, output_filename="data.txt"):
    with open(output_filename, "w") as f_w:
        for text in dataset["text"]:
            f_w.write(text+"\n")