import json
import os
from datasets import load_dataset


def prepare_arc_dataset():
  os.makedirs("data/sampled", exist_ok=True)

  print("Fetching ARC-Challenge from Hugging Face...")
  arc_validation = load_dataset(
      "allenai/ai2_arc", "ARC-Challenge", split="validation"
  )
  arc_test = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

  dev_sample = arc_validation.select(range(100))
  test_sample = arc_test.select(range(300))

  def format_split(hf_split):
    formatted = []
    for row in hf_split:
      formatted.append({
          "id": row["id"],
          "question": row["question"],
          "choices": {
              "labels": row["choices"]["label"],
              "texts": row["choices"]["text"],
          },
          "answerKey": row["answerKey"],
      })
    return formatted

  with open("data/sampled/dev_100.jsonl", "w", encoding="utf-8") as f:
    for item in format_split(dev_sample):
      f.write(json.dumps(item) + "\n")

  with open("data/sampled/test_300.jsonl", "w", encoding="utf-8") as f:
    for item in format_split(test_sample):
      f.write(json.dumps(item) + "\n")

  print("Done! Files saved in 'data/sampled/':")
  print(" - data/sampled/dev_100.jsonl (100 questions)")
  print(" - data/sampled/test_300.jsonl (300 questions)")


if __name__ == "__main__":
  prepare_arc_dataset()