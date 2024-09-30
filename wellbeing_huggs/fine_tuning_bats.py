import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from transformers import DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split

def load_prepare_dataset(split_ratio=0.9):
    dataset = load_dataset('Vedant64/counsel_chat')

    # Combine questions and answers to make full conversational texts
    texts = [(q if q is not None else "") + " " + (a if a is not None else "") for q, a in zip(dataset['train']['questionTitle'], dataset['train']['answerText'])]

    # Splitting the dataset into training and validation sets
    train_texts, val_texts = train_test_split(texts, test_size=(1 - split_ratio))
    
    # Returning a dictionary format suitable for fine-tuning
    train_dataset = Dataset.from_dict({'text': train_texts})
    val_dataset = Dataset.from_dict(({'text': val_texts}))
    
    return train_dataset, val_dataset

# Tokenizing data
def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# Fine-tuning Process
def fine_tune_gpt2(train_dataset, val_dataset, tokenizer, model_name, output_dir, epochs=3):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Data Collator for Language Modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Tokenize the dataset
    train_dataset = train_dataset.map(lambda e: tokenize_function(e, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda e: tokenize_function(e, tokenizer), batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,           # output directory
        num_train_epochs=epochs,         # total number of training epochs
        per_device_train_batch_size=8,   # batch size for training
        per_device_eval_batch_size=8,    # batch size for evaluation
        logging_dir='./logs',            # directory for storing logs
        logging_steps=200,               # log every 200 steps
        save_total_limit=3,              # limit the total amount of checkpoints
        evaluation_strategy="epoch",     # evaluate at the end of every epoch
        save_strategy="epoch",           # save model at the end of every epoch
        load_best_model_at_end=True,     # load the best model when finished
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Training model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save metrics to JSON files
    with open(f'{output_dir}/training_metrics.json', 'w') as f:
        json.dump(trainer.state.log_history, f)

    return model, trainer

if __name__ == "__main__":
    dataset_name = "counsel_chat"  # CounselChat dataset
    model_name = "gpt2"  # GPT-2 model to fine-tune
    output_dir = "./fine_tuned_model"  # Directory to save the fine-tuned model and tokenizer
    
    # Load the tokenizer and dataset
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Handle the padding token error
    
    # Load and prepare CounselChat dataset
    train_dataset, val_dataset = load_prepare_dataset()

    # Fine-tune the model
    model, trainer = fine_tune_gpt2(train_dataset, val_dataset, tokenizer, model_name, output_dir, epochs=3)

    # Save evaluation metrics
    metrics = trainer.evaluate()

    with open(f'{output_dir}/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f)

    print(f"Fine-tuned model, tokenizer, and metrics saved to {output_dir}")